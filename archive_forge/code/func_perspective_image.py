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
@_register_kernel_internal(perspective, torch.Tensor)
@_register_kernel_internal(perspective, tv_tensors.Image)
def perspective_image(image: torch.Tensor, startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, fill: _FillTypeJIT=None, coefficients: Optional[List[float]]=None) -> torch.Tensor:
    perspective_coeffs = _perspective_coefficients(startpoints, endpoints, coefficients)
    interpolation = _check_interpolation(interpolation)
    if image.numel() == 0:
        return image
    shape = image.shape
    ndim = image.ndim
    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False
    _assert_grid_transform_inputs(image, matrix=None, interpolation=interpolation.value, fill=fill, supported_interpolation_modes=['nearest', 'bilinear'], coeffs=perspective_coeffs)
    oh, ow = shape[-2:]
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    grid = _perspective_grid(perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=image.device)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)
    if needs_unsquash:
        output = output.reshape(shape)
    return output