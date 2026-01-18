import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def row_separator(self, overall_width):
    return [f'{self._num_threads} threads: '.ljust(overall_width, '-')] if self._num_threads is not None else []