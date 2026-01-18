from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def pts_to_distogram(pts: torch.Tensor, min_bin: torch.types.Number=2.3125, max_bin: torch.types.Number=21.6875, no_bins: int=64) -> torch.Tensor:
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries)