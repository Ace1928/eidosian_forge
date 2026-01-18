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
def p_sizeof(s):
    pos = s.position()
    s.next()
    s.expect('(')
    if looking_at_expr(s):
        operand = p_test(s)
        node = ExprNodes.SizeofVarNode(pos, operand=operand)
    else:
        base_type = p_c_base_type(s)
        declarator = p_c_declarator(s, empty=1)
        node = ExprNodes.SizeofTypeNode(pos, base_type=base_type, declarator=declarator)
    s.expect(')')
    return node