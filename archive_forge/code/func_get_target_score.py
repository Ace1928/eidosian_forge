from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
@staticmethod
def get_target_score(i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, full_precision: bool, margin: float, scale: Optional[float]) -> torch.Tensor:
    tokens, d_model = i.shape
    assert d_model == w.shape[1]
    tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
    assert tw.shape == (tokens, d_model)
    if scale is not None:
        target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
    else:
        target_score = i * tw
    if full_precision:
        target_score = target_score.float()
    target_score = target_score.sum(dim=1)
    if scale is not None:
        target_score -= margin
        target_score *= scale
    return target_score