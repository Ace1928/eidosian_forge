from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
@_register_kernel_internal(get_size, torch.Tensor)
@_register_kernel_internal(get_size, tv_tensors.Image, tv_tensor_wrapper=False)
def get_size_image(image: torch.Tensor) -> List[int]:
    hw = list(image.shape[-2:])
    ndims = len(hw)
    if ndims == 2:
        return hw
    else:
        raise TypeError(f'Input tensor should have at least two dimensions, but got {ndims}')