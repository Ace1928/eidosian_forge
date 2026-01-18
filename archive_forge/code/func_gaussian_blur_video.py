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
@_register_kernel_internal(gaussian_blur, tv_tensors.Video)
def gaussian_blur_video(video: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]]=None) -> torch.Tensor:
    return gaussian_blur_image(video, kernel_size, sigma)