from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def get_num_frames(inpt: torch.Tensor) -> int:
    if torch.jit.is_scripting():
        return get_num_frames_video(inpt)
    _log_api_usage_once(get_num_frames)
    kernel = _get_kernel(get_num_frames, type(inpt))
    return kernel(inpt)