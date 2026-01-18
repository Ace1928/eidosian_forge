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
def p_fused_definition(s, pos, ctx):
    """
    c(type)def fused my_fused_type:
        ...
    """
    if ctx.level not in ('module', 'module_pxd'):
        error(pos, 'Fused type definition not allowed here')
    s.next()
    name = p_ident(s)
    s.expect(':')
    s.expect_newline()
    s.expect_indent()
    types = []
    while s.sy != 'DEDENT':
        if s.sy != 'pass':
            types.append(p_c_base_type(s))
        else:
            s.next()
        s.expect_newline()
    s.expect_dedent()
    if not types:
        error(pos, 'Need at least one type')
    return Nodes.FusedTypeNode(pos, name=name, types=types)