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
@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround=(), exclusive_arg_names=()) -> PatternExpr:
    argnames = [*inspect.signature(search_fn).parameters.keys()]
    if scalar_workaround == ():
        scalar_workaround = {}
    flat_inputs = []
    input_idx = 0
    for argname in argnames:
        if argname in scalar_workaround:
            flat_inputs.append(scalar_workaround[argname])
        else:
            flat_inputs.append(example_inputs[input_idx])
            input_idx += 1
    search_gm = trace_fn(search_fn, flat_inputs)
    return fx_to_pattern(search_gm, ignore_types=(int, float, list, torch.device, torch.dtype), argnames=argnames, scalar_workaround=scalar_workaround, exclusive_arg_names=exclusive_arg_names)