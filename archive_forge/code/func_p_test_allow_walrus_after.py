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
def p_test_allow_walrus_after(s):
    if s.sy == 'lambda':
        return p_lambdef(s)
    pos = s.position()
    expr = p_or_test(s)
    if s.sy == 'if':
        s.next()
        test = p_or_test(s)
        s.expect('else')
        other = p_test(s)
        return ExprNodes.CondExprNode(pos, test=test, true_val=expr, false_val=other)
    else:
        return expr