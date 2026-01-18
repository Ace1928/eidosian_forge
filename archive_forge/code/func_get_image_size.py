import warnings
from typing import Any, List
import torch
from torchvision.transforms import functional as _F
def get_image_size(inpt: torch.Tensor) -> List[int]:
    warnings.warn('The function `get_image_size(...)` is deprecated and will be removed in a future release. Instead, please use `get_size(...)` which returns `[h, w]` instead of `[w, h]`.')
    return _F.get_image_size(inpt)