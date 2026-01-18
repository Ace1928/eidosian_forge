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
@torch.enable_grad()
def joint_fwd_bwd(fn, args) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: Optional[torch.fx.GraphModule] = None

    def record_joint_graph(joint_graph, inputs, **kwargs):
        nonlocal gm
        assert not gm
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)
    with torch._guards.tracing(None):
        aot_function(fn, lambda g, i: make_boxed_func(g), partition_fn=record_joint_graph, decompositions=select_decomp_table(), keep_inference_input_mutations=True, enable_log=False)(*args)
    assert gm
    from .fx_passes.joint_graph import pointless_view
    matcher_pass = PatternMatcherPass()
    pattern = CallFunction(torch.ops.aten.view.default, KeywordArg('arg'), KeywordArg('size'))
    GraphPatternEntry(pattern=pattern, handler=pointless_view, extra_check=_return_true).register(matcher_pass.patterns)
    matcher_pass.apply(gm.graph)
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm