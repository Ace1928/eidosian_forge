from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def merge_et_kernel_metadata(lhs: Dict[str, List[str]], rhs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merge_et_kernel_metadata: Dict[str, Set[str]] = defaultdict(set)
    for op in list(lhs.keys()) + list(rhs.keys()):
        merge_et_kernel_metadata[op].update(lhs.get(op, []))
        merge_et_kernel_metadata[op].update(rhs.get(op, []))
    return {op: sorted(val) for op, val in merge_et_kernel_metadata.items()}