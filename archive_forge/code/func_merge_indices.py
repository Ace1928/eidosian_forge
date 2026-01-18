import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
@staticmethod
def merge_indices(index_a: 'ETKernelIndex', index_b: 'ETKernelIndex') -> 'ETKernelIndex':
    combined = defaultdict(dict, index_a.index.copy())
    for op, entry in index_b.index.items():
        for key, metadata in entry.items():
            combined[op][key] = metadata
    return ETKernelIndex(combined)