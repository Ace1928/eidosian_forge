import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def highlight_warnings(self):
    self._highlight_warnings = True