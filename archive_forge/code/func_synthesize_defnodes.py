from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def synthesize_defnodes(self):
    """
        Create the __signatures__ dict of PyCFunctionNode specializations.
        """
    if isinstance(self.nodes[0], CFuncDefNode):
        nodes = [node.py_func for node in self.nodes]
    else:
        nodes = self.nodes
    for node in nodes:
        node.entry.signature.use_fastcall = False
    signatures = [StringEncoding.EncodedString(node.specialized_signature_string) for node in nodes]
    keys = [ExprNodes.StringNode(node.pos, value=sig) for node, sig in zip(nodes, signatures)]
    values = [ExprNodes.PyCFunctionNode.from_defnode(node, binding=True) for node in nodes]
    self.__signatures__ = ExprNodes.DictNode.from_pairs(self.pos, zip(keys, values))
    self.specialized_pycfuncs = values
    for pycfuncnode in values:
        pycfuncnode.is_specialization = True