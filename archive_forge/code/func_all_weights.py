import math
import warnings
import numbers
import weakref
from typing import List, Tuple, Optional, overload
import torch
from torch import Tensor
from .module import Module
from ..parameter import Parameter
from ..utils.rnn import PackedSequence
from .. import init
from ... import _VF
@property
def all_weights(self) -> List[List[Parameter]]:
    return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]