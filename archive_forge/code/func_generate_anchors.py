import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList
def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype: torch.dtype=torch.float32, device: torch.device=torch.device('cpu')) -> Tensor:
    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
    return base_anchors.round()