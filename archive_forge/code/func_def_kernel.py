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
def def_kernel(self, inputs: List[IRNode], outputs: List[IRNode], names_str: str='', input_reorder: Optional[List[int]]=None) -> str:
    """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
        """
    names = [x.strip() for x in names_str.strip().split(',')]
    if len(inputs) + len(outputs) != len(names):
        raise RuntimeError(f'len(inputs) + len(outputs)={len(inputs) + len(outputs)!r} != len(names)={len(names)!r}, inputs={inputs!r}, outputs={outputs!r}, names={names!r}')
    if input_reorder is not None:
        assert len(inputs) == len(input_reorder)
    else:
        input_reorder = list(range(len(inputs)))
    for idx in input_reorder:
        name = names[idx]
        node = inputs[idx]
        if node is not None:
            self.named_nodes[name] = node
            self.args.input_buffers[node.get_name()] = name
    for name, node in zip(names[len(inputs):len(inputs) + len(outputs)], outputs):
        if node is not None:
            self.named_nodes[name] = node
            self.args.output_buffers[node.get_name()] = name
    arg_defs, *_ = self.args.cpp_argdefs()
    return f'PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})'