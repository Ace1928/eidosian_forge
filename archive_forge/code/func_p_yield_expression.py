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
def p_yield_expression(s):
    pos = s.position()
    s.next()
    is_yield_from = False
    if s.sy == 'from':
        is_yield_from = True
        s.next()
    if s.sy != ')' and s.sy not in statement_terminators:
        arg = p_test(s) if is_yield_from else p_testlist(s)
    else:
        if is_yield_from:
            s.error("'yield from' requires a source argument", pos=pos, fatal=False)
        arg = None
    if is_yield_from:
        return ExprNodes.YieldFromExprNode(pos, arg=arg)
    else:
        return ExprNodes.YieldExprNode(pos, arg=arg)