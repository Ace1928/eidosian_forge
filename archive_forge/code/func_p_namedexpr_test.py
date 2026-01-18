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
def p_namedexpr_test(s):
    lhs = p_test_allow_walrus_after(s)
    if s.sy == ':=':
        position = s.position()
        if not lhs.is_name:
            s.error('Left-hand side of assignment expression must be an identifier', fatal=False)
        s.next()
        rhs = p_test(s)
        return ExprNodes.AssignmentExpressionNode(position, lhs=lhs, rhs=rhs)
    return lhs