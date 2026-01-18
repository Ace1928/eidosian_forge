import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def detach_and_move(t: Tensor, to_cpu: bool) -> Tensor:
    t = t.detach()
    if to_cpu:
        t = t.cpu()
    return t