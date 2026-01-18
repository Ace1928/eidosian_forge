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
def resized_crop_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, top: int, left: int, height: int, width: int, size: List[int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bounding_boxes, canvas_size = crop_bounding_boxes(bounding_boxes, format, top, left, height, width)
    return resize_bounding_boxes(bounding_boxes, canvas_size=canvas_size, size=size)