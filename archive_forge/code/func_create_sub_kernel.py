import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sympy import Integer
from .. import metrics
from ..scheduler import SchedulerNode
from ..utils import ceildiv, Placeholder
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta
def create_sub_kernel(self, *groups, index_dtype, mutations, reduction_hint):
    sub_kernel = TritonKernel(*groups, index_dtype=index_dtype, mutations=mutations, pid_cache={'tl.program_id(0)': 'xpid_offset', 'tl.program_id(1)': 'ypid'}, reduction_hint=reduction_hint)
    if self.blocking_2d:
        assert len(groups) == 3
    self.blocking_2d |= groups[1] != 1 and len(groups) == 3
    metrics.generated_kernel_count -= 1
    sub_kernel.args = self.args
    sub_kernel.iter_vars_count = self.iter_vars_count
    sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
    self.sub_kernels.append(sub_kernel)
    return sub_kernel