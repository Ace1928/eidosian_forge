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
def p_DEF_statement(s):
    pos = s.position()
    denv = s.compile_time_env
    s.next()
    name = p_ident(s)
    s.expect('=')
    expr = p_compile_time_expr(s)
    if s.compile_time_eval:
        value = expr.compile_time_value(denv)
        denv.declare(name, value)
    s.expect_newline('Expected a newline', ignore_semicolon=True)
    return Nodes.PassStatNode(pos)