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
@_register_kernel_internal(rotate, torch.Tensor)
@_register_kernel_internal(rotate, tv_tensors.Image)
def rotate_image(image: torch.Tensor, angle: float, interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, expand: bool=False, center: Optional[List[float]]=None, fill: _FillTypeJIT=None) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)
    shape = image.shape
    num_channels, height, width = shape[-3:]
    center_f = [0.0, 0.0]
    if center is not None:
        center_f = [c - s * 0.5 for c, s in zip(center, [width, height])]
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    if image.numel() > 0:
        image = image.reshape(-1, num_channels, height, width)
        _assert_grid_transform_inputs(image, matrix, interpolation.value, fill, ['nearest', 'bilinear'])
        ow, oh = _compute_affine_output_size(matrix, width, height) if expand else (width, height)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _affine_grid(theta, w=width, h=height, ow=ow, oh=oh)
        output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)
        new_height, new_width = output.shape[-2:]
    else:
        output = image
        new_width, new_height = _compute_affine_output_size(matrix, width, height) if expand else (width, height)
    return output.reshape(shape[:-3] + (num_channels, new_height, new_width))