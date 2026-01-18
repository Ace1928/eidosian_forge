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
def p_power(s):
    if s.systring == 'new' and s.peek()[0] == 'IDENT':
        return p_new_expr(s)
    await_pos = None
    if s.sy == 'await':
        await_pos = s.position()
        s.next()
    n1 = p_atom(s)
    while s.sy in ('(', '[', '.'):
        n1 = p_trailer(s, n1)
    if await_pos:
        n1 = ExprNodes.AwaitExprNode(await_pos, arg=n1)
    if s.sy == '**':
        pos = s.position()
        s.next()
        n2 = p_factor(s)
        n1 = ExprNodes.binop_node(pos, '**', n1, n2)
    return n1