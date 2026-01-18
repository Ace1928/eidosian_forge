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
@_register_kernel_internal(gaussian_blur, torch.Tensor)
@_register_kernel_internal(gaussian_blur, tv_tensors.Image)
def gaussian_blur_image(image: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]]=None) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(f'If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}')
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f'kernel_size should have odd and positive integers. Got {kernel_size}')
    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
    elif isinstance(sigma, (list, tuple)):
        length = len(sigma)
        if length == 1:
            s = float(sigma[0])
            sigma = [s, s]
        elif length != 2:
            raise ValueError(f'If sigma is a sequence, its length should be 2. Got {length}')
    elif isinstance(sigma, (int, float)):
        s = float(sigma)
        sigma = [s, s]
    else:
        raise TypeError(f'sigma should be either float or sequence of floats. Got {type(sigma)}')
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f'sigma should have positive values. Got {sigma}')
    if image.numel() == 0:
        return image
    dtype = image.dtype
    shape = image.shape
    ndim = image.ndim
    if ndim == 3:
        image = image.unsqueeze(dim=0)
    elif ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
    fp = torch.is_floating_point(image)
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype if fp else torch.float32, device=image.device)
    kernel = kernel.expand(shape[-3], 1, kernel.shape[0], kernel.shape[1])
    output = image if fp else image.to(dtype=torch.float32)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    output = torch_pad(output, padding, mode='reflect')
    output = conv2d(output, kernel, groups=shape[-3])
    if ndim == 3:
        output = output.squeeze(dim=0)
    elif ndim > 4:
        output = output.reshape(shape)
    if not fp:
        output = output.round_().to(dtype=dtype)
    return output