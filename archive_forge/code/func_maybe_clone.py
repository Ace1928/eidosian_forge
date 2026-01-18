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
def maybe_clone(t):
    if isinstance(t, torch.Tensor) and StorageWeakRef(t._typed_storage()) in input_storage:
        return t.clone()
    return t