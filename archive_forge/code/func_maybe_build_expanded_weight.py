import functools
import torch
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight
from torch.utils import _pytree as pytree
def maybe_build_expanded_weight(og_tensor, batch_size):
    if og_tensor.requires_grad:
        return ExpandedWeight(og_tensor, batch_size, loss_reduction)
    else:
        return og_tensor