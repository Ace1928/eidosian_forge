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
def p_IF_statement(s, ctx):
    pos = s.position()
    saved_eval = s.compile_time_eval
    current_eval = saved_eval
    denv = s.compile_time_env
    result = None
    while 1:
        s.next()
        expr = p_compile_time_expr(s)
        s.compile_time_eval = current_eval and bool(expr.compile_time_value(denv))
        body = p_suite(s, ctx)
        if s.compile_time_eval:
            result = body
            current_eval = 0
        if s.sy != 'ELIF':
            break
    if s.sy == 'ELSE':
        s.next()
        s.compile_time_eval = current_eval
        body = p_suite(s, ctx)
        if current_eval:
            result = body
    if not result:
        result = Nodes.PassStatNode(pos)
    s.compile_time_eval = saved_eval
    return result