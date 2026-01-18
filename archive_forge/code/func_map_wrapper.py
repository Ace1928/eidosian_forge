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
def map_wrapper(f, xs, *args):
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    if not all((isinstance(t, torch.Tensor) for t in flat_xs)):
        raise RuntimeError(f'Mapped xs can only consist of tensors. Got xs {flat_xs}.')
    num_mapped_args = len(flat_xs)
    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError('Leading dimensions of mapped xs cannot be 0.')
    if any((cur_shape[0] != leading_dim_size for cur_shape in shapes)):
        raise RuntimeError(f'Leading dimensions of mapped xs must be consistent. Got shapes {shapes}.')
    out_spec = None

    def flat_fn(*flat_args):
        xs = pytree.tree_unflatten(flat_args[:num_mapped_args], xs_spec)
        unflattened_out = f(xs, *flat_args[num_mapped_args:])
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)
        nonlocal out_spec
        out_spec = tmp_out_spec
        return flat_out
    return pytree.tree_unflatten(map_impl(flat_fn, num_mapped_args, *flat_xs, *args), out_spec)