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
def p_cascaded_cmp(s):
    pos = s.position()
    op = p_cmp_op(s)
    n2 = p_starred_expr(s)
    result = ExprNodes.CascadedCmpNode(pos, operator=op, operand2=n2)
    if s.sy in comparison_ops:
        result.cascade = p_cascaded_cmp(s)
    return result