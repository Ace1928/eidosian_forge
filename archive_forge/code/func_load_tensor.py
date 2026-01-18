import contextlib
from typing import Sequence
import torch
from torch._custom_op.impl import custom_op
from torch.utils._content_store import ContentStoreReader
@custom_op('debugprims::load_tensor')
def load_tensor(name: str, size: Sequence[int], stride: Sequence[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ...