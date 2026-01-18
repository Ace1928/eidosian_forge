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
def p_testlist_comp(s):
    pos = s.position()
    expr = p_namedexpr_test_or_starred_expr(s)
    if s.sy == ',':
        s.next()
        exprs = p_namedexpr_test_or_starred_expr_list(s, expr)
        return ExprNodes.TupleNode(pos, args=exprs)
    elif s.sy in ('for', 'async'):
        return p_genexp(s, expr)
    else:
        return expr