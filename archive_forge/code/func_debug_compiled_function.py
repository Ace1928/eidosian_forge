import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .runtime_wrappers import (
from .schemas import (
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import (
@wraps(compiled_function)
def debug_compiled_function(*args):
    for i, a in enumerate(args):
        can_require_grad = flat_requires_grad[i]
        if can_require_grad is None:
            assert not isinstance(a, Tensor)
        elif not can_require_grad:
            assert not a.requires_grad, format_guard_bug_msg(aot_config, f'{describe_input(i, aot_config)} would not require grad')
    return compiled_function(*args)