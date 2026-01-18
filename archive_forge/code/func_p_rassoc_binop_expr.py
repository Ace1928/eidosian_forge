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
def p_rassoc_binop_expr(s, op, p_subexpr):
    n1 = p_subexpr(s)
    if s.sy == op:
        pos = s.position()
        op = s.sy
        s.next()
        n2 = p_rassoc_binop_expr(s, op, p_subexpr)
        n1 = ExprNodes.binop_node(pos, op, n1, n2)
    elif s.sy in COMMON_BINOP_MISTAKES and COMMON_BINOP_MISTAKES[s.sy] == op:
        warning(s.position(), "Found the C operator '%s', did you mean the Python operator '%s'?" % (s.sy, op), level=1)
    return n1