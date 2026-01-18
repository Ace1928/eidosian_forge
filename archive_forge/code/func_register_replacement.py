from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
def register_replacement(search_fn, replace_fn, example_inputs: Iterable[Any], trace_fn: Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule], pass_dicts, extra_check=_return_true, scalar_workaround=(), exclusive_arg_names=(), search_fn_pattern=None):
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint forward+backward graph.

    Args:
        search_fn: traced to give original pattern
        replace_fn: traced to give replacement graph
        example_inputs: example inputs for initial trace
        trace_fn: fwd_only or joint_fwd_bwd
        pass_dict: dict of passes to register to
        extra_check: additional check to run on match(using real shapes)
    """
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    def check_fn(match: Match):
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.

        Recheck the match with the correct shapes.
        """
        for name in argnames:
            if name not in match.kwargs:
                raise RuntimeError(f'Not all inputs to pattern found in match.kwargs. Perhaps one of the inputs is unused? argnames={argnames}, match.kwargs={match.kwargs}')
        args = list(torch.fx.map_arg([match.kwargs[name] for name in argnames], lambda n: n.meta['val']))
        with torch._dynamo.utils.detect_fake_mode(args):
            for i, grad in enumerate(requires_grad):
                if isinstance(args[i], torch.Tensor):
                    if grad and is_integer_dtype(args[i].dtype):
                        return False
                    args[i] = torch.empty_strided(args[i].size(), args[i].stride(), dtype=args[i].dtype, device=args[i].device, requires_grad=grad)
            try:
                specific_graph = trace_fn(search_fn, args)
            except RuntimeError as e:
                log.info('Replacement pattern %s failed to apply due to shape mismatch: %s', search_fn.__name__, e)
                return False
            specific_pattern = fx_to_pattern(specific_graph, argnames=argnames, exclusive_arg_names=exclusive_arg_names, scalar_workaround=scalar_workaround)
            specific_pattern_match = specific_pattern.match(match.output_nodes()[0])
            if specific_pattern_match and extra_check(specific_pattern_match):
                match.replacement_graph = trace_fn(replace_fn, args)
                return True
            return False

    def normalize_args(**kwargs):
        args = []
        for name in argnames:
            args.append(kwargs.pop(name))
        for i in range(1, len(kwargs) + 1):
            if f'tangents_{i}' not in kwargs:
                break
            args.append(kwargs.pop(f'tangents_{i}'))
        assert not kwargs, f'leftover kwargs: {kwargs!r}'
        return args
    if trace_fn is joint_fwd_bwd:
        if torch.is_inference_mode_enabled():
            return False
    with functorch_config.patch(functionalize_rng_ops=False):
        requires_grad: List[bool] = [isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs]
        if search_fn_pattern is None:
            pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround, exclusive_arg_names)
        else:
            pattern = search_fn_pattern
        pattern_repr = PatternPrettyPrinter.run(pattern)
        assert pattern_repr not in _seen_patterns
        _seen_patterns.add(pattern_repr)
        pattern = ReplacementPatternEntry(pattern=pattern, extra_check=check_fn, normalize_args=normalize_args)
        pattern.register(pass_dicts)
        return pattern.pattern