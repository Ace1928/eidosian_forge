from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def convert_to_ufunc(node):
    if isinstance(node, Nodes.CFuncDefNode):
        if node.local_scope.parent_scope.is_c_class_scope:
            error(node.pos, 'Methods cannot currently be converted to a ufunc')
            return node
        converters = [UFuncConversion(node)]
        original_node = node
    elif isinstance(node, FusedNode.FusedCFuncDefNode) and isinstance(node.node, Nodes.CFuncDefNode):
        if node.node.local_scope.parent_scope.is_c_class_scope:
            error(node.pos, 'Methods cannot currently be converted to a ufunc')
            return node
        converters = [UFuncConversion(n) for n in node.nodes]
        original_node = node.node
    else:
        error(node.pos, 'Only C functions can be converted to a ufunc')
        return node
    if not converters:
        return
    del converters[0].global_scope.entries[original_node.entry.name]
    converters[0].use_generic_utility_code()
    return [node] + _generate_stats_from_converters(converters, original_node)