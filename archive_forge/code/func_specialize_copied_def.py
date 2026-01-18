from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def specialize_copied_def(self, node, cname, py_entry, f2s, fused_compound_types):
    """Specialize the copy of a DefNode given the copied node,
        the specialization cname and the original DefNode entry"""
    fused_types = self._get_fused_base_types(fused_compound_types)
    type_strings = [PyrexTypes.specialization_signature_string(fused_type, f2s) for fused_type in fused_types]
    node.specialized_signature_string = '|'.join(type_strings)
    node.entry.pymethdef_cname = PyrexTypes.get_fused_cname(cname, node.entry.pymethdef_cname)
    node.entry.doc = py_entry.doc
    node.entry.doc_cname = py_entry.doc_cname