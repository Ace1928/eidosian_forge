import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
def pin_memory(self):
    return type(self)(self.data.pin_memory(), self.batch_sizes, bind(self.sorted_indices, lambda t: t.pin_memory()), bind(self.unsorted_indices, lambda t: t.pin_memory()))