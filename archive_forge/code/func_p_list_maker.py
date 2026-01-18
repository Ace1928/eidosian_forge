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
def p_list_maker(s):
    pos = s.position()
    s.next()
    if s.sy == ']':
        s.expect(']')
        return ExprNodes.ListNode(pos, args=[])
    expr = p_namedexpr_test_or_starred_expr(s)
    if s.sy in ('for', 'async'):
        if expr.is_starred:
            s.error('iterable unpacking cannot be used in comprehension')
        append = ExprNodes.ComprehensionAppendNode(pos, expr=expr)
        loop = p_comp_for(s, append)
        s.expect(']')
        return ExprNodes.ComprehensionNode(pos, loop=loop, append=append, type=Builtin.list_type, has_local_scope=s.context.language_level >= 3)
    if s.sy == ',':
        s.next()
        exprs = p_namedexpr_test_or_starred_expr_list(s, expr)
    else:
        exprs = [expr]
    s.expect(']')
    return ExprNodes.ListNode(pos, args=exprs)