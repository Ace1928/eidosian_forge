from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def init_gpu_tensors(self, rank: int) -> None:
    assert self.num_accepted_tokens is None
    device = f'cuda:{rank}'
    self.num_accepted_tokens = torch.tensor(0, dtype=torch.long, device=device)
    self.num_emitted_tokens = torch.tensor(0, dtype=torch.long, device=device)