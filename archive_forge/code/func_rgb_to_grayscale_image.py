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
@_register_kernel_internal(rgb_to_grayscale, torch.Tensor)
@_register_kernel_internal(rgb_to_grayscale, tv_tensors.Image)
def rgb_to_grayscale_image(image: torch.Tensor, num_output_channels: int=1) -> torch.Tensor:
    if num_output_channels not in (1, 3):
        raise ValueError(f'num_output_channels must be 1 or 3, got {num_output_channels}.')
    return _rgb_to_grayscale_image(image, num_output_channels=num_output_channels, preserve_dtype=True)