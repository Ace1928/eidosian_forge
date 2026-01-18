from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def lmcl_pre_softmax(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = F.normalize(input, dim=1)
    w = F.normalize(self.fc.weight, dim=1)
    logits = torch.einsum('nc,kc->nk', x, w)
    row_ind = torch.arange(x.shape[0], dtype=torch.long).to(x.device)
    col_ind = target
    logits[row_ind, col_ind] -= self.margin
    logits *= self.scale
    return logits