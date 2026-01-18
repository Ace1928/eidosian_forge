from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
@staticmethod
def ir_to_evt_string(template_output_node_name: str, evt_type_name: str, epilogue_nodes: List[IRNode]):
    """
        Formats IR nodes into a string representation compatible with Cutlass EVT format.

        Args:
            template_output_node_name (str): The name of the template output node.
            evt_type_name (str): The name of the EVT type.
            epilogue_nodes (List[IRNode]): A list of IR nodes representing the epilogue nodes. As of now, these must be
                ComputedBuffer nodes wrapping Pointwise nodes.

        Returns:
            A string representation of the IR nodes formatted according to the Cutlass EVT format.
        """
    formatter = CutlassEVTEpilogueTypeFormatter(template_output_node_name, evt_type_name)
    with virtualized.V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
        for node in epilogue_nodes:
            if isinstance(node, ComputedBuffer):
                pnode = node.data
            else:
                raise RuntimeError('Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer')
            assert isinstance(pnode, Pointwise)
            index = pnode._index(pnode.ranges)
            result = pnode.inner_fn(index)
            formatter.aliases[node.name] = result
        res = formatter.getvalue(result)
        if _MAGIC_SYMPY_ERROR_STRING in res:
            raise CUTLASSEVTOpNotImplementedError('sympy / indexing expressions not yet supported in EVT fusion')
        else:
            return res