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
def p_c_struct_or_union_definition(s, pos, ctx):
    packed = False
    if s.systring == 'packed':
        packed = True
        s.next()
        if s.sy != 'IDENT' or s.systring != 'struct':
            s.expected('struct')
    kind = s.systring
    s.next()
    name = p_ident(s)
    cname = p_opt_cname(s)
    if cname is None and ctx.namespace is not None:
        cname = ctx.namespace + '::' + name
    attributes = None
    if s.sy == ':':
        s.next()
        attributes = []
        if s.sy == 'pass':
            s.next()
            s.expect_newline('Expected a newline', ignore_semicolon=True)
        else:
            s.expect('NEWLINE')
            s.expect_indent()
            body_ctx = Ctx(visibility=ctx.visibility)
            while s.sy != 'DEDENT':
                if s.sy != 'pass':
                    attributes.append(p_c_func_or_var_declaration(s, s.position(), body_ctx))
                else:
                    s.next()
                    s.expect_newline('Expected a newline')
            s.expect_dedent()
        if not attributes and ctx.visibility != 'extern':
            error(pos, "Empty struct or union definition not allowed outside a 'cdef extern from' block")
    else:
        s.expect_newline('Syntax error in struct or union definition')
    return Nodes.CStructOrUnionDefNode(pos, name=name, cname=cname, kind=kind, attributes=attributes, typedef_flag=ctx.typedef_flag, visibility=ctx.visibility, api=ctx.api, in_pxd=ctx.level == 'module_pxd', packed=packed)