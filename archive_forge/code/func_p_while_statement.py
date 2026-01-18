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
def p_while_statement(s):
    pos = s.position()
    s.next()
    test = p_namedexpr_test(s)
    body = p_suite(s)
    else_clause = p_else_clause(s)
    return Nodes.WhileStatNode(pos, condition=test, body=body, else_clause=else_clause)