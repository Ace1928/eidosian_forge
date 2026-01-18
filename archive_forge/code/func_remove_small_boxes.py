from typing import Tuple
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
from ._utils import _upcast
def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than min_size
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(remove_small_boxes)
    ws, hs = (boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep