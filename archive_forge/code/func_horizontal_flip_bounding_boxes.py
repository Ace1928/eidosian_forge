import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def horizontal_flip_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int]) -> torch.Tensor:
    shape = bounding_boxes.shape
    bounding_boxes = bounding_boxes.clone().reshape(-1, 4)
    if format == tv_tensors.BoundingBoxFormat.XYXY:
        bounding_boxes[:, [2, 0]] = bounding_boxes[:, [0, 2]].sub_(canvas_size[1]).neg_()
    elif format == tv_tensors.BoundingBoxFormat.XYWH:
        bounding_boxes[:, 0].add_(bounding_boxes[:, 2]).sub_(canvas_size[1]).neg_()
    else:
        bounding_boxes[:, 0].sub_(canvas_size[1]).neg_()
    return bounding_boxes.reshape(shape)