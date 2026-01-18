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
@_register_kernel_internal(elastic, torch.Tensor)
@_register_kernel_internal(elastic, tv_tensors.Image)
def elastic_image(image: torch.Tensor, displacement: torch.Tensor, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, fill: _FillTypeJIT=None) -> torch.Tensor:
    if not isinstance(displacement, torch.Tensor):
        raise TypeError('Argument displacement should be a Tensor')
    interpolation = _check_interpolation(interpolation)
    if image.numel() == 0:
        return image
    shape = image.shape
    ndim = image.ndim
    device = image.device
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    is_cpu_half = device.type == 'cpu' and dtype == torch.float16
    if is_cpu_half:
        image = image.to(torch.float32)
        dtype = torch.float32
    expected_shape = (1,) + shape[-2:] + (2,)
    if expected_shape != displacement.shape:
        raise ValueError(f'Argument displacement shape should be {expected_shape}, but given {displacement.shape}')
    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False
    if displacement.dtype != dtype or displacement.device != device:
        displacement = displacement.to(dtype=dtype, device=device)
    image_height, image_width = shape[-2:]
    grid = _create_identity_grid((image_height, image_width), device=device, dtype=dtype).add_(displacement)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)
    if needs_unsquash:
        output = output.reshape(shape)
    if is_cpu_half:
        output = output.to(torch.float16)
    return output