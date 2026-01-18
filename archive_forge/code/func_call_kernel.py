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
def call_kernel(self, name: str, node: 'CUDATemplateBuffer', epilogue_nodes: List[ir.Buffer]) -> None:
    """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen

        name: Name of kernel function.
        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
    wrapper = V.graph.wrapper_code
    _, call_args, _ = self.args.python_argdefs()
    for i in range(len(call_args)):
        if V.graph.is_unspec_arg(call_args[i]):
            call_args[i] = call_args[i] + '.item()'
        else:
            call_args[i] = f'c_void_p({call_args[i]}.data_ptr())'
    call_args.append('None')
    if node.get_workspace_size() > 0:
        call_args.append(f'c_void_p({node.get_name()}_workspace.data_ptr())')
    else:
        call_args.append('None')
    wrapper.generate_kernel_call(name, call_args, device_index=V.graph.scheduler.current_device.index, cuda=True, triton=False)