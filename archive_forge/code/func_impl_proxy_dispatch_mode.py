from typing import Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.types import _device, _dtype
@run_with_rng_state.py_impl(ProxyTorchDispatchMode)
def impl_proxy_dispatch_mode(mode, rng_state, op, *args, **kwargs):
    if mode.enable_tracing:
        with disable_proxy_modes_tracing():
            out = run_with_rng_state(rng_state, op, *args, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (rng_state, op, *args))
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy('call_function', run_with_rng_state, proxy_args, proxy_kwargs)
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    else:
        return run_with_rng_state(rng_state, op, *args, **kwargs)