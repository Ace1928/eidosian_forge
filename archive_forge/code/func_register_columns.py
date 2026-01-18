import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def register_columns(self, columns: Tuple[_Column, ...]):
    self._columns = columns