import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def extract_buffers(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    return _extract_members(mod, mod.named_buffers, lambda x: x)