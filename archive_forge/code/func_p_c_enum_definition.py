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
def p_c_enum_definition(s, pos, ctx):
    s.next()
    scoped = False
    if s.context.cpp and (s.sy == 'class' or (s.sy == 'IDENT' and s.systring == 'struct')):
        scoped = True
        s.next()
    if s.sy == 'IDENT':
        name = s.systring
        s.next()
        cname = p_opt_cname(s)
        if cname is None and ctx.namespace is not None:
            cname = ctx.namespace + '::' + name
    else:
        name = cname = None
        if scoped:
            s.error('Unnamed scoped enum not allowed')
    if scoped and s.sy == '(':
        s.next()
        underlying_type = p_c_base_type(s)
        s.expect(')')
    else:
        underlying_type = Nodes.CSimpleBaseTypeNode(pos, name='int', module_path=[], is_basic_c_type=True, signed=1, complex=0, longness=0)
    s.expect(':')
    items = []
    doc = None
    if s.sy != 'NEWLINE':
        p_c_enum_line(s, ctx, items)
    else:
        s.next()
        s.expect_indent()
        doc = p_doc_string(s)
        while s.sy not in ('DEDENT', 'EOF'):
            p_c_enum_line(s, ctx, items)
        s.expect_dedent()
    if not items and ctx.visibility != 'extern':
        error(pos, "Empty enum definition not allowed outside a 'cdef extern from' block")
    return Nodes.CEnumDefNode(pos, name=name, cname=cname, scoped=scoped, items=items, underlying_type=underlying_type, typedef_flag=ctx.typedef_flag, visibility=ctx.visibility, create_wrapper=ctx.overridable, api=ctx.api, in_pxd=ctx.level == 'module_pxd', doc=doc)