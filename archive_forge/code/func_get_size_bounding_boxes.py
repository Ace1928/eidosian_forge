from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
@_register_kernel_internal(get_size, tv_tensors.BoundingBoxes, tv_tensor_wrapper=False)
def get_size_bounding_boxes(bounding_box: tv_tensors.BoundingBoxes) -> List[int]:
    return list(bounding_box.canvas_size)