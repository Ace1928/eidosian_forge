import math
from dataclasses import dataclass
from typing import (
import torch
def split_kv(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    return self.k_seqinfo.split(tensor, self._batch_sizes)