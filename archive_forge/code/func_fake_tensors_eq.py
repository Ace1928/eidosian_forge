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
def fake_tensors_eq(t1, t2, fields=('shape', 'dtype', 'device')):
    if any((not isinstance(t, torch.Tensor) for t in (t1, t2))):
        return False
    for field in fields:
        if getattr(t1, field) != getattr(t2, field):
            return False
    return True