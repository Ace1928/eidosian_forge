import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
from .make_fx import randomize
import re
def func_no_tensors(args):
    reconstructed_flat_args = []
    args = iter(args)
    for v in flat_args:
        if isinstance(v, torch.Tensor):
            reconstructed_flat_args.append(next(args))
        else:
            reconstructed_flat_args.append(v)
    c_args, c_kwargs = pytree.tree_unflatten(reconstructed_flat_args, args_spec)
    return func(*c_args, **c_kwargs)