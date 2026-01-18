from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def eval_forward(self, input: torch.Tensor) -> torch.Tensor:
    """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
    return torch.matmul(input, self.proj_weight.T)