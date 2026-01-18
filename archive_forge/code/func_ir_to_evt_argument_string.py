from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
@staticmethod
def ir_to_evt_argument_string(template_output_node_name: str, epilogue_nodes: List[IRNode]) -> str:
    formatter = CutlassEVTEpilogueArgumentFormatter(template_output_node_name)
    with virtualized.V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
        for node in epilogue_nodes:
            assert isinstance(node, ComputedBuffer)
            pnode = node.data
            assert isinstance(pnode, Pointwise)
            index = pnode._index(pnode.ranges)
            result = pnode.inner_fn(index)
            if node.name is not None:
                formatter.aliases[node.name] = result
        res: str = formatter.getvalue(result)
        if _MAGIC_SYMPY_ERROR_STRING in res:
            raise CUTLASSEVTOpNotImplementedError('sympy / indexing expressions not yet supported in EVT fusion')
        else:
            return res