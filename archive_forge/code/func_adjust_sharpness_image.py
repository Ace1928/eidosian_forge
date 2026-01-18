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
@_register_kernel_internal(adjust_sharpness, torch.Tensor)
@_register_kernel_internal(adjust_sharpness, tv_tensors.Image)
def adjust_sharpness_image(image: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    num_channels, height, width = image.shape[-3:]
    if num_channels not in (1, 3):
        raise TypeError(f'Input image tensor can have 1 or 3 channels, but found {num_channels}')
    if sharpness_factor < 0:
        raise ValueError(f'sharpness_factor ({sharpness_factor}) is not non-negative.')
    if image.numel() == 0 or height <= 2 or width <= 2:
        return image
    bound = _max_value(image.dtype)
    fp = image.is_floating_point()
    shape = image.shape
    if image.ndim > 4:
        image = image.reshape(-1, num_channels, height, width)
        needs_unsquash = True
    else:
        needs_unsquash = False
    kernel_dtype = image.dtype if fp else torch.float32
    a, b = (1.0 / 13.0, 5.0 / 13.0)
    kernel = torch.tensor([[a, a, a], [a, b, a], [a, a, a]], dtype=kernel_dtype, device=image.device)
    kernel = kernel.expand(num_channels, 1, 3, 3)
    output = image.to(dtype=kernel_dtype, copy=True)
    blurred_degenerate = conv2d(output, kernel, groups=num_channels)
    if not fp:
        blurred_degenerate = blurred_degenerate.round_()
    view = output[..., 1:-1, 1:-1]
    view.add_(blurred_degenerate.sub_(view), alpha=1.0 - sharpness_factor)
    output = output.clamp_(0, bound)
    if not fp:
        output = output.to(image.dtype)
    if needs_unsquash:
        output = output.reshape(shape)
    return output