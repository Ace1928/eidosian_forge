import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
@staticmethod
def col_fn(m: common.Measurement) -> Optional[str]:
    return m.description