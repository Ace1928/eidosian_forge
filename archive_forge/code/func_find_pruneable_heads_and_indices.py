import inspect
from typing import Callable, List, Optional, Set, Tuple, Union
import torch
from packaging import version
from safetensors.torch import storage_ptr, storage_size
from torch import nn
from .utils import is_torch_tpu_available, logging
def find_pruneable_heads_and_indices(heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum((1 if h < head else 0 for h in already_pruned_heads))
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return (heads, index)