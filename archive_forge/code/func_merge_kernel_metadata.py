from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def merge_kernel_metadata(lhs: Dict[str, List[str]], rhs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    kernel_metadata: Dict[str, List[str]] = {}
    for tag_name, dtypes in list(lhs.items()) + list(rhs.items()):
        dtypes_copy = set(dtypes)
        if tag_name in kernel_metadata:
            dtypes_copy |= set(kernel_metadata[tag_name])
        kernel_metadata[tag_name] = list(dtypes_copy)
    return kernel_metadata