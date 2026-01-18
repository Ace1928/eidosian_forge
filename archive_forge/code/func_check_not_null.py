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
def check_not_null(self, node: IRNode) -> str:
    """
        Generates code to check that a node is not null.
        """
    if node is None:
        return ''
    size_str = self.size(node, 0, -1)
    name_str = self.arg_name(node)
    if name_str is None:
        return ''
    res = IndentedBuffer(initial_indent=2)
    res.tabwidth = 1
    res.splice(f'\n            {{\n              if (!{name_str}) {{\n                int64_t {name_str}_size = {size_str};\n                if ({name_str}_size > 0) {{\n                  throw std::runtime_error("input {name_str} is null but size is not 0!");\n                }}\n              }}\n            }}\n            ')
    return res.getvalue()