import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
@staticmethod
def grow_from_backend_indices(kernel_index: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]], backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]) -> None:
    for dk in backend_indices:
        index = backend_indices[dk]
        for op, backend_metadata in index.items():
            if op in kernel_index:
                kernel_index[op][ETKernelKey(default=True)] = backend_metadata
            else:
                kernel_index[op] = {ETKernelKey(default=True): backend_metadata}