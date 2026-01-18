import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [torch.tensor(s, dtype=torch.float32, device=keypoints.device) / torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device) for s, s_orig in zip(new_size, original_size)]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data