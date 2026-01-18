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
@_register_kernel_internal(adjust_saturation, torch.Tensor)
@_register_kernel_internal(adjust_saturation, tv_tensors.Image)
def adjust_saturation_image(image: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    if saturation_factor < 0:
        raise ValueError(f'saturation_factor ({saturation_factor}) is not non-negative.')
    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f'Input image tensor permitted channel values are 1 or 3, but found {c}')
    if c == 1:
        return image
    grayscale_image = _rgb_to_grayscale_image(image, num_output_channels=1, preserve_dtype=False)
    if not image.is_floating_point():
        grayscale_image = grayscale_image.floor_()
    return _blend(image, grayscale_image, saturation_factor)