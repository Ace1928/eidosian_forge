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
def p_comparison(s):
    n1 = p_starred_expr(s)
    if s.sy in comparison_ops:
        pos = s.position()
        op = p_cmp_op(s)
        n2 = p_starred_expr(s)
        n1 = ExprNodes.PrimaryCmpNode(pos, operator=op, operand1=n1, operand2=n2)
        if s.sy in comparison_ops:
            n1.cascade = p_cascaded_cmp(s)
    return n1