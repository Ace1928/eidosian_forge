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
def p_ctypedef_statement(s, ctx):
    pos = s.position()
    s.next()
    visibility = p_visibility(s, ctx.visibility)
    api = p_api(s)
    ctx = ctx(typedef_flag=1, visibility=visibility)
    if api:
        ctx.api = 1
    if s.sy == 'class':
        return p_c_class_definition(s, pos, ctx)
    elif s.sy == 'IDENT' and s.systring in struct_enum_union:
        return p_struct_enum(s, pos, ctx)
    elif s.sy == 'IDENT' and s.systring == 'fused':
        return p_fused_definition(s, pos, ctx)
    else:
        base_type = p_c_base_type(s, nonempty=1)
        declarator = p_c_declarator(s, ctx, is_type=1, nonempty=1)
        s.expect_newline('Syntax error in ctypedef statement', ignore_semicolon=True)
        return Nodes.CTypeDefNode(pos, base_type=base_type, declarator=declarator, visibility=visibility, api=api, in_pxd=ctx.level == 'module_pxd')