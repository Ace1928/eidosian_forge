from typing import List
import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value
from torchvision.utils import _log_api_usage_once
from ._misc import _num_value_bits, to_dtype_image
from ._type_conversion import pil_to_tensor, to_pil_image
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(adjust_hue, torch.Tensor)
@_register_kernel_internal(adjust_hue, tv_tensors.Image)
def adjust_hue_image(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(f'hue_factor ({hue_factor}) is not in [-0.5, 0.5].')
    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f'Input image tensor permitted channel values are 1 or 3, but found {c}')
    if c == 1:
        return image
    if image.numel() == 0:
        return image
    orig_dtype = image.dtype
    image = to_dtype_image(image, torch.float32, scale=True)
    image = _rgb_to_hsv(image)
    h, s, v = image.unbind(dim=-3)
    h.add_(hue_factor).remainder_(1.0)
    image = torch.stack((h, s, v), dim=-3)
    image_hue_adj = _hsv_to_rgb(image)
    return to_dtype_image(image_hue_adj, orig_dtype, scale=True)