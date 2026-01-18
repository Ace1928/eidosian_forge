from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, fn):
    import torch
    assert len(args) == 1
    grad = args[0]
    assert isinstance(grad, torch.Tensor)

    def self_invoke(*args):
        return _trace_wrapped_op(*args, fn=fn)
    proxy_args = (mode.tracer.unwrap_proxy(grad),)
    out_proxy = mode.tracer.create_proxy('call_function', self_invoke, proxy_args, {}, name='trace_wrapped')
    grad = torch.zeros_like(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    proxy_args = (mode.tracer.unwrap_proxy(grad), grad.size(), grad.stride(), grad.dtype)
    out_proxy = mode.tracer.create_proxy('call_function', _assert_meta, proxy_args, {}, name='assert')
    grad = torch.empty_like(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad