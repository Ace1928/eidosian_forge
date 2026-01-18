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
@wraps(compiled_fw)
def rng_functionalization_wrapper(args):
    if fw_metadata.is_rng_op_functionalized:
        seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
        args.extend([seed, offset])
        out = compiled_fw(args)
        out = functionalized_rng_runtime_epilogue(fw_metadata, out)
        return out
    else:
        return compiled_fw(args)