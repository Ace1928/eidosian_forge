import warnings
from contextlib import nullcontext
from typing import Any, Callable, List, Tuple, Union
from unittest.mock import patch
import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx import Interpreter
from torch.fx.experimental.symbolic_shapes import definitely_false, sym_eq
from torch.nn.utils import stateless
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import (
from .logging_utils import setup_stacktrace_preservation_hooks
from .schemas import (
from .subclass_utils import (
from .utils import maybe_to_fresh_input
def create_joint(fn: Callable, *, aot_config: AOTConfig) -> Any:

    def inner_fn(primals: List[Any], tangents: List[Any]):
        outs, tangent_mask = fn(*primals)
        assert len(tangent_mask) == len(outs)
        outs_to_grad = [o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent]
        assert len(outs_to_grad) == len(tangents)
        grad_primals = []
        inputs_needs_grads = []
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs_to_grad, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                needed_outs.append(out if not definitely_false(sym_eq(out.shape, tangent.shape)) else out.view(tangent.shape))
                needed_tangents.append(tangent)
        setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])
        if config.functionalize_rng_ops:
            PhiloxStateTracker.mark_beginning_of_backward()
        backward_out: Tuple[Tensor, ...] = tuple()
        if grad_primals:
            with fx_traceback.preserve_node_meta():
                if aot_config.no_tangents:
                    assert len(needed_tangents) == 1 and needed_tangents[0].numel() == 1
                    backward_out = torch.autograd.grad(needed_outs, grad_primals, allow_unused=True)
                else:
                    backward_out = torch.autograd.grad(needed_outs, grad_primals, grad_outputs=needed_tangents, allow_unused=True)
        backward_out_iter = iter(backward_out)
        return (outs, [next(backward_out_iter) if i else None for i in inputs_needs_grads])

    def inner_fn_with_anomaly(*args):
        with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Anomaly Detection has been enabled.')
            with torch.autograd.detect_anomaly(check_nan=False):
                return inner_fn(*args)
    return inner_fn_with_anomaly