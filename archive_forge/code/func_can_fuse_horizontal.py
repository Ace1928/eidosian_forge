from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling
def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    for node in (node1, node2):
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(node) or self._cuda_cpp_scheduling.is_cuda_cpp_fused_template(node):
            return self._cuda_cpp_scheduling.can_fuse_horizontal(node1, node2)
    return self._triton_scheduling.can_fuse_horizontal(node1, node2)