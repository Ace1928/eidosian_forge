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
def register_run_with_rng_state_op():
    run_with_rng_state = HigherOrderOperator('run_with_rng_state')
    run_with_rng_state.py_impl(DispatchKey.Autograd)(autograd_not_implemented(run_with_rng_state, deferred_error=True))

    @run_with_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(rng_state, op, *args, **kwargs):
        current_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state.cpu())
        out = op(*args, **kwargs)
        torch.cuda.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(rng_state, op, *args, **kwargs):
        current_state = torch.get_rng_state()
        torch.set_rng_state(rng_state)
        out = op(*args, **kwargs)
        torch.set_rng_state(current_state)
        return out

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

    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        impl_map = {'cuda': impl_cuda, 'cpu': impl_cpu}
        device = get_device(args, kwargs)
        assert device in impl_map, f'Backend not supported for {device}'
        impl = impl_map[device]
        return impl(rng_state, op, *args, **kwargs)

    @run_with_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, rng_state, op, *args, **kwargs):
        with mode:
            return op(*args, **kwargs)
    return run_with_rng_state