import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint
from torch._functorch.eager_transforms import (
from torch._higher_order_ops.cond import (
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, num_mapped, *args):
    mode = _get_current_dispatch_mode()
    assert mode is not None, 'Mode should always be enabled for python fallback key'
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_map(mode, map_impl, f, num_mapped, *args)
        else:
            return map_impl(f, num_mapped, *args)