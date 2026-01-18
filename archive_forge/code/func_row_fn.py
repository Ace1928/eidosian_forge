import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
@staticmethod
def row_fn(m: common.Measurement) -> Tuple[int, Optional[str], str]:
    return (m.num_threads, m.env, m.as_row_name)