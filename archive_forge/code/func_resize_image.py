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
@_register_kernel_internal(resize, torch.Tensor)
@_register_kernel_internal(resize, tv_tensors.Image)
def resize_image(image: torch.Tensor, size: List[int], interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, max_size: Optional[int]=None, antialias: Optional[Union[str, bool]]='warn') -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)
    antialias = _check_antialias(img=image, antialias=antialias, interpolation=interpolation)
    assert not isinstance(antialias, str)
    antialias = False if antialias is None else antialias
    align_corners: Optional[bool] = None
    if interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC:
        align_corners = False
    else:
        antialias = False
    shape = image.shape
    numel = image.numel()
    num_channels, old_height, old_width = shape[-3:]
    new_height, new_width = _compute_resized_output_size((old_height, old_width), size=size, max_size=max_size)
    if (new_height, new_width) == (old_height, old_width):
        return image
    elif numel > 0:
        image = image.reshape(-1, num_channels, old_height, old_width)
        dtype = image.dtype
        acceptable_dtypes = [torch.float32, torch.float64]
        if interpolation == InterpolationMode.NEAREST or interpolation == InterpolationMode.NEAREST_EXACT:
            acceptable_dtypes.append(torch.uint8)
        elif image.device.type == 'cpu':
            if interpolation == InterpolationMode.BILINEAR and 'AVX2' in torch.backends.cpu.get_cpu_capability() or interpolation == InterpolationMode.BICUBIC:
                acceptable_dtypes.append(torch.uint8)
        strides = image.stride()
        if image.is_contiguous(memory_format=torch.channels_last) and image.shape[0] == 1 and (numel != strides[0]):
            new_strides = list(strides)
            new_strides[0] = numel
            image = image.as_strided((1, num_channels, old_height, old_width), new_strides)
        need_cast = dtype not in acceptable_dtypes
        if need_cast:
            image = image.to(dtype=torch.float32)
        image = interpolate(image, size=[new_height, new_width], mode=interpolation.value, align_corners=align_corners, antialias=antialias)
        if need_cast:
            if interpolation == InterpolationMode.BICUBIC and dtype == torch.uint8:
                image = image.clamp_(min=0, max=255)
            if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                image = image.round_()
            image = image.to(dtype=dtype)
    return image.reshape(shape[:-3] + (num_channels, new_height, new_width))