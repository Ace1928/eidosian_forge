from contextlib import contextmanager
import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(mode, *args, **kwargs):
    if not mode.enable_tracing:
        return _export_tracepoint(*args, **kwargs)
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    proxy = mode.tracer.create_proxy('call_function', _export_tracepoint, p_args, p_kwargs)
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)