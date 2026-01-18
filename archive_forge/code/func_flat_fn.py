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
def flat_fn(*flat_args):
    xs = pytree.tree_unflatten(flat_args[:num_mapped_args], xs_spec)
    unflattened_out = f(xs, *flat_args[num_mapped_args:])
    flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)
    nonlocal out_spec
    out_spec = tmp_out_spec
    return flat_out