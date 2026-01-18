from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def lmcl_matmul(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, w_idx: int, margin: float, scale: Optional[float]) -> torch.Tensor:
    """LMCL variation of matmul with normalization, margin and scale."""
    logits = torch.matmul(F.normalize(i, dim=1), F.normalize(w, dim=1).T)
    mask = torch.arange(w_idx * w.shape[0], (w_idx + 1) * w.shape[0], dtype=torch.long, device=i.device).expand(i.shape[0], -1)
    logits[mask == tgt.reshape(-1, 1)] -= margin
    logits *= scale
    return logits