import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
from .make_fx import randomize
import re
def call_forwards_backwards(f, args):
    flat_args = pytree.arg_tree_leaves(*args)
    diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and arg.requires_grad]
    out = wrapper_set_seed(f, args)
    flat_out = pytree.tree_leaves(out)
    sm = 0
    for i in flat_out:
        if isinstance(i, torch.Tensor):
            sm += i.sum().abs()
    assert isinstance(sm, torch.Tensor)
    return (out, torch.autograd.grad(sm, diff_args, allow_unused=True))