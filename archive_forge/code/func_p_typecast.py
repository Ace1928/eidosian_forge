from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_typecast(s):
    pos = s.position()
    s.next()
    base_type = p_c_base_type(s)
    is_memslice = isinstance(base_type, Nodes.MemoryViewSliceTypeNode)
    is_other_unnamed_type = isinstance(base_type, (Nodes.TemplatedTypeNode, Nodes.CConstOrVolatileTypeNode, Nodes.CTupleBaseTypeNode))
    if not (is_memslice or is_other_unnamed_type) and base_type.name is None:
        s.error('Unknown type')
    declarator = p_c_declarator(s, empty=1)
    if s.sy == '?':
        s.next()
        typecheck = 1
    else:
        typecheck = 0
    s.expect('>')
    operand = p_factor(s)
    if is_memslice:
        return ExprNodes.CythonArrayNode(pos, base_type_node=base_type, operand=operand)
    return ExprNodes.TypecastNode(pos, base_type=base_type, declarator=declarator, operand=operand, typecheck=typecheck)