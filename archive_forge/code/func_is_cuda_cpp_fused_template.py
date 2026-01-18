import logging
from typing import cast, List
from ...._dynamo.utils import counters
from ... import config, ir
from ...codecache import code_hash, get_path
from ...ir import ComputedBuffer, CUDATemplateBuffer, Pointwise
from ...scheduler import (
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from .cutlass_epilogue_gen import CUTLASSEVTOpNotImplementedError
def is_cuda_cpp_fused_template(self, node: BaseSchedulerNode) -> bool:
    return isinstance(node, FusedSchedulerNode) and self.is_cuda_cpp_template(node.get_template_node())