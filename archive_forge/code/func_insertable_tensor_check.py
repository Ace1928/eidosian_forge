import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
def insertable_tensor_check(self, t: torch.Tensor) -> bool:
    return t.numel() != 0 and bool((t == t.flatten()[0]).all()) and torch._C._has_storage(t) and (t.layout == torch.strided)