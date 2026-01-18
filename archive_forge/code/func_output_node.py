import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from ... import ir
from ...autotune_process import CUDABenchmarkRequest
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V
from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP
def output_node(self) -> TensorBox:
    return TensorBox.create(CUDATemplateBuffer(layout=self.layout, inputs=self.input_nodes, make_kernel_render=self.make_kernel_render, workspace_size=self.bmreq.workspace_size, template=self.template))