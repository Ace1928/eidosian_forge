from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
    if self.include_all_operators or self.include_all_non_op_selectives:
        return True
    return kernel_tag in self.kernel_metadata and dtype in self.kernel_metadata[kernel_tag]