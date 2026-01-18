from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def create_new_local_scope(self, node, env, f2s):
    """
        Create a new local scope for the copied node and append it to
        self.nodes. A new local scope is needed because the arguments with the
        fused types are already in the local scope, and we need the specialized
        entries created after analyse_declarations on each specialized version
        of the (CFunc)DefNode.
        f2s is a dict mapping each fused type to its specialized version
        """
    node.create_local_scope(env)
    node.local_scope.fused_to_specific = f2s
    node.has_fused_arguments = False
    self.nodes.append(node)