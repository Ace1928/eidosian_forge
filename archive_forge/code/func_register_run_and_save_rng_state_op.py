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
def register_run_and_save_rng_state_op():
    run_and_save_rng_state = HigherOrderOperator('run_and_save_rng_state')
    run_and_save_rng_state.py_impl(DispatchKey.Autograd)(autograd_not_implemented(run_and_save_rng_state, deferred_error=True))

    @run_and_save_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, **kwargs):
        return (torch.cuda.get_rng_state(), op(*args, **kwargs))

    @run_and_save_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(op, *args, **kwargs):
        return (torch.get_rng_state(), op(*args, **kwargs))

    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        impl_map = {'cuda': impl_cuda, 'cpu': impl_cpu}
        device = get_device(args, kwargs)
        assert device in impl_map, f'Backend not supported for {device}'
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, op, *args, **kwargs):
        with mode:
            return impl_backend_select(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, **kwargs):
        if mode.enable_tracing:
            out = impl_backend_select(op, *args, **kwargs)
            proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
            proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
            out_proxy = mode.tracer.create_proxy('call_function', run_and_save_rng_state, proxy_args, proxy_kwargs)
            return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        else:
            return run_and_save_rng_state(op, *args, **kwargs)
    return run_and_save_rng_state