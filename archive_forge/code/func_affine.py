import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def affine(inpt: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, fill: _FillTypeJIT=None, center: Optional[List[float]]=None) -> torch.Tensor:
    """[BETA] See :class:`~torchvision.transforms.v2.RandomAffine` for details."""
    if torch.jit.is_scripting():
        return affine_image(inpt, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, center=center)
    _log_api_usage_once(affine)
    kernel = _get_kernel(affine, type(inpt))
    return kernel(inpt, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=interpolation, fill=fill, center=center)