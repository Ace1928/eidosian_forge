import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def padded_3d(tensors: List[torch.LongTensor], pad_idx: int=0, use_cuda: bool=False, dtype: Optional[torch.dtype]=torch.long, fp16friendly: bool=False):
    """
    Make 3D padded tensor for list of lists of 1D tensors or lists.

    :param tensors:
        list of lists of 1D tensors (or lists)
    :param pad_idx:
        padding to fill tensor with
    :param use_cuda:
        whether to call cuda() before returning
    :param bool fp16friendly:
        if True, pads the final dimension to be a multiple of 8.

    :returns:
        3D tensor with the maximum dimensions of the inputs
    """
    a = len(tensors)
    b = max((len(row) for row in tensors))
    c = max((len(item) for row in tensors for item in row))
    if fp16friendly and c % FP16_PAD_SIZE != 0:
        c += FP16_PAD_SIZE - c % FP16_PAD_SIZE
    c = max(c, 1)
    output = torch.full((a, b, c), pad_idx, dtype=dtype)
    for i, row in enumerate(tensors):
        item: Sized
        for j, item in enumerate(row):
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.Tensor(item, dtype=dtype)
            output[i, j, :len(item)] = item
    if use_cuda:
        output = output.cuda()
    return output