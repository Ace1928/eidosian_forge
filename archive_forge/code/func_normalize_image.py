import math
from typing import List, Optional
import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms._functional_tensor import _max_value
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(normalize, torch.Tensor)
@_register_kernel_internal(normalize, tv_tensors.Image)
def normalize_image(image: torch.Tensor, mean: List[float], std: List[float], inplace: bool=False) -> torch.Tensor:
    if not image.is_floating_point():
        raise TypeError(f'Input tensor should be a float tensor. Got {image.dtype}.')
    if image.ndim < 3:
        raise ValueError(f'Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}.')
    if isinstance(std, (tuple, list)):
        divzero = not all(std)
    elif isinstance(std, (int, float)):
        divzero = std == 0
    else:
        divzero = False
    if divzero:
        raise ValueError('std evaluated to zero, leading to division by zero.')
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    if inplace:
        image = image.sub_(mean)
    else:
        image = image.sub(mean)
    return image.div_(std)