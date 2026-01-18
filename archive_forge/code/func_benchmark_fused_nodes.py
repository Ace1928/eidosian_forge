from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling
def benchmark_fused_nodes(self, nodes):
    return self._triton_scheduling.benchmark_fused_nodes(nodes)