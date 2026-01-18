from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def replace_fused_typechecks(self, copied_node):
    """
        Branch-prune fused type checks like

            if fused_t is int:
                ...

        Returns whether an error was issued and whether we should stop in
        in order to prevent a flood of errors.
        """
    num_errors = Errors.get_errors_count()
    transform = ParseTreeTransforms.ReplaceFusedTypeChecks(copied_node.local_scope)
    transform(copied_node)
    if Errors.get_errors_count() > num_errors:
        return False
    return True