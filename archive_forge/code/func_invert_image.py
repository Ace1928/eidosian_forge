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
@_register_kernel_internal(invert, torch.Tensor)
@_register_kernel_internal(invert, tv_tensors.Image)
def invert_image(image: torch.Tensor) -> torch.Tensor:
    if image.is_floating_point():
        return 1.0 - image
    elif image.dtype == torch.uint8:
        return image.bitwise_not()
    else:
        return image.bitwise_xor((1 << _num_value_bits(image.dtype)) - 1)