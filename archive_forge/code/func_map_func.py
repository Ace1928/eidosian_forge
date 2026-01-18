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
@map_impl.py_impl(DispatchKey.Functionalize)
def map_func(f, num_mapped, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(pos_args, reapply_views=reapply_views)
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        functional_map_fn = functionalize(f, remove=mode)
        with disable_proxy_modes_tracing():
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        if _has_potential_branch_input_mutation(f, example_inputs):
            raise UnsupportedAliasMutationException('torch.map is mutating the input!')
        if _has_potential_branch_input_alias(f, example_inputs):
            raise UnsupportedAliasMutationException('torch.map is aliasing the input!')
        map_return = map_impl(functional_map_fn, num_mapped, *unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=0)