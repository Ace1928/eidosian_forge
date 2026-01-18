import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
@staticmethod
def from_backend_indices(backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]) -> 'ETKernelIndex':
    kernel_index: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] = defaultdict(dict)
    ETKernelIndex.grow_from_backend_indices(kernel_index, backend_indices)
    return ETKernelIndex(kernel_index)