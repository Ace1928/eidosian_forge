import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@property
def symints_saved_for_backwards_slice(self):
    assert self.num_symints_saved_for_bw is not None
    if self.num_symints_saved_for_bw > 0:
        return slice(-self.num_symints_saved_for_bw, None)
    else:
        return slice(0, 0)