import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [torch.tensor(s, dtype=torch.float32, device=boxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)