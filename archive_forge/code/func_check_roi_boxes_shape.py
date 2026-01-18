from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(_tensor.size(1) == 4, 'The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]')
    elif isinstance(boxes, torch.Tensor):
        torch._assert(boxes.size(1) == 5, 'The boxes tensor shape is not correct as Tensor[K, 5]')
    else:
        torch._assert(False, 'boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]')
    return