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
def row_or_column_stride(self, node: IRNode, default_value: int=0) -> str:
    """
        Hook called from template code to get the row or column stride of an arg.
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
        """
    if node is None or len(node.get_stride()) < 2:
        return str(default_value)
    stride0 = node.get_stride()[-1]
    stride1 = node.get_stride()[-2]
    if stride0 == 1:
        return cexpr(self.rename_indexing(stride1))
    elif stride1 == 1:
        return cexpr(self.rename_indexing(stride0))
    else:
        raise RuntimeError(f'At least 1 stride should be 1. Strides: node.get_stride()={node.get_stride()!r}')