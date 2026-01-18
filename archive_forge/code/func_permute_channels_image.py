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
@_register_kernel_internal(permute_channels, torch.Tensor)
@_register_kernel_internal(permute_channels, tv_tensors.Image)
def permute_channels_image(image: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    shape = image.shape
    num_channels, height, width = shape[-3:]
    if len(permutation) != num_channels:
        raise ValueError(f'Length of permutation does not match number of channels: {len(permutation)} != {num_channels}')
    if image.numel() == 0:
        return image
    image = image.reshape(-1, num_channels, height, width)
    image = image[:, permutation, :, :]
    return image.reshape(shape)