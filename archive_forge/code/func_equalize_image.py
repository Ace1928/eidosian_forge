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
@_register_kernel_internal(equalize, torch.Tensor)
@_register_kernel_internal(equalize, tv_tensors.Image)
def equalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.numel() == 0:
        return image
    output_dtype = image.dtype
    image = to_dtype_image(image, torch.uint8, scale=True)
    batch_shape = image.shape[:-2]
    flat_image = image.flatten(start_dim=-2).to(torch.long)
    hist = flat_image.new_zeros(batch_shape + (256,), dtype=torch.int32)
    hist.scatter_add_(dim=-1, index=flat_image, src=hist.new_ones(1).expand_as(flat_image))
    cum_hist = hist.cumsum(dim=-1)
    index = cum_hist.argmax(dim=-1)
    num_non_max_pixels = flat_image.shape[-1] - hist.gather(dim=-1, index=index.unsqueeze_(-1))
    step = num_non_max_pixels.div_(255, rounding_mode='floor')
    valid_equalization = step.ne(0).unsqueeze_(-1)
    cum_hist = cum_hist[..., :-1]
    cum_hist.add_(step // 2).div_(step.clamp_(min=1), rounding_mode='floor').clamp_(0, 255)
    lut = cum_hist.to(torch.uint8)
    lut = torch.cat([lut.new_zeros(1).expand(batch_shape + (1,)), lut], dim=-1)
    equalized_image = lut.gather(dim=-1, index=flat_image).view_as(image)
    output = torch.where(valid_equalization, equalized_image, image)
    return to_dtype_image(output, output_dtype, scale=True)