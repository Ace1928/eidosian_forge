from typing import Tuple
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
from ._utils import _upcast
def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Converts boxes from given in_fmt to out_fmt.
    Supported in_fmt and out_fmt are:

    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(box_convert)
    allowed_fmts = ('xyxy', 'xywh', 'cxcywh')
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError('Unsupported Bounding Box Conversions for given in_fmt and out_fmt')
    if in_fmt == out_fmt:
        return boxes.clone()
    if in_fmt != 'xyxy' and out_fmt != 'xyxy':
        if in_fmt == 'xywh':
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == 'cxcywh':
            boxes = _box_cxcywh_to_xyxy(boxes)
        in_fmt = 'xyxy'
    if in_fmt == 'xyxy':
        if out_fmt == 'xywh':
            boxes = _box_xyxy_to_xywh(boxes)
        elif out_fmt == 'cxcywh':
            boxes = _box_xyxy_to_cxcywh(boxes)
    elif out_fmt == 'xyxy':
        if in_fmt == 'xywh':
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == 'cxcywh':
            boxes = _box_cxcywh_to_xyxy(boxes)
    return boxes