import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
import torch.utils._pytree as pytree
def randomize(args):

    def transform(x):
        if not x.dtype.is_floating_point:
            return x
        return x.detach().clone().uniform_(0, 1).requires_grad_(x.requires_grad)
    return pytree.tree_map_only(torch.Tensor, transform, args)