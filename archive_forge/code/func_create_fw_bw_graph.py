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
def create_fw_bw_graph(f, num_mapped_args, *args):
    mapped_xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]
    with suspend_functionalization():
        with disable_proxy_modes_tracing():

            def from_fun(t):
                if isinstance(t, torch.Tensor):
                    if t.dtype != torch.bool:
                        return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, requires_grad=t.requires_grad)
                    else:
                        return t.clone()
                return t
            example_xs = [from_fun(xs) for xs in _unstack_pytree(mapped_xs)[0]]
            example_pos_args = [from_fun(arg) if isinstance(arg, torch.Tensor) else arg for arg in pos_args]
            example_flat_out = pytree.tree_map(from_fun, f(*example_xs, *example_pos_args))
            if any((not isinstance(out, torch.Tensor) for out in example_flat_out if out is not None)):
                raise RuntimeError(f'Expect outputs of map only contains tensors or None. Got types {[type(out) for out in example_flat_out]}.')
            example_grad = [from_fun(out) for out in example_flat_out]
            fw_graph = make_fx(f)(*example_xs, *example_pos_args)

        def joint_f(*example_args):
            joint_mapped_args = example_args[:joint_num_mapped]
            args = example_args[joint_num_mapped:]
            mapped_input = joint_mapped_args[:num_mapped_args]
            mapped_grads = joint_mapped_args[num_mapped_args:]

            def fw_with_masks(*args):
                fw_out = f(*args)
                return (fw_out, [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in fw_out])
            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            _, grads = joint(list(mapped_input) + list(args), [grad for grad in mapped_grads if grad is not None and grad.requires_grad])
            input_storage = {StorageWeakRef(arg._typed_storage()) for arg in example_args if isinstance(arg, torch.Tensor)}

            def maybe_clone(t):
                if isinstance(t, torch.Tensor) and StorageWeakRef(t._typed_storage()) in input_storage:
                    return t.clone()
                return t
            return pytree.tree_map(maybe_clone, grads)
        joint_num_mapped = len(example_grad) + len(example_xs)
        joint_graph = make_fx(joint_f)(*example_xs, *example_grad, *example_pos_args)
        return (fw_graph, joint_graph)