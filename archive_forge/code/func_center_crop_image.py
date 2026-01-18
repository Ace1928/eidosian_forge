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
@_register_kernel_internal(center_crop, torch.Tensor)
@_register_kernel_internal(center_crop, tv_tensors.Image)
def center_crop_image(image: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    shape = image.shape
    if image.numel() == 0:
        return image.reshape(shape[:-2] + (crop_height, crop_width))
    image_height, image_width = shape[-2:]
    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        image = torch_pad(image, _parse_pad_padding(padding_ltrb), value=0.0)
        image_height, image_width = image.shape[-2:]
        if crop_width == image_width and crop_height == image_height:
            return image
    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return image[..., crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]