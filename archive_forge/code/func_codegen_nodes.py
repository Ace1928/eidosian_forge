from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling
def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
    return self._triton_scheduling.codegen_nodes(nodes)