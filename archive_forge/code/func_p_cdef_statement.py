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
def p_cdef_statement(s, ctx):
    pos = s.position()
    ctx.visibility = p_visibility(s, ctx.visibility)
    ctx.api = ctx.api or p_api(s)
    if ctx.api:
        if ctx.visibility not in ('private', 'public'):
            error(pos, "Cannot combine 'api' with '%s'" % ctx.visibility)
    if ctx.visibility == 'extern' and s.sy == 'from':
        return p_cdef_extern_block(s, pos, ctx)
    elif s.sy == 'import':
        s.next()
        return p_cdef_extern_block(s, pos, ctx)
    elif p_nogil(s):
        ctx.nogil = 1
        if ctx.overridable:
            error(pos, 'cdef blocks cannot be declared cpdef')
        return p_cdef_block(s, ctx)
    elif s.sy == ':':
        if ctx.overridable:
            error(pos, 'cdef blocks cannot be declared cpdef')
        return p_cdef_block(s, ctx)
    elif s.sy == 'class':
        if ctx.level not in ('module', 'module_pxd'):
            error(pos, 'Extension type definition not allowed here')
        if ctx.overridable:
            error(pos, 'Extension types cannot be declared cpdef')
        return p_c_class_definition(s, pos, ctx)
    elif s.sy == 'IDENT' and s.systring == 'cppclass':
        return p_cpp_class_definition(s, pos, ctx)
    elif s.sy == 'IDENT' and s.systring in struct_enum_union:
        if ctx.level not in ('module', 'module_pxd'):
            error(pos, 'C struct/union/enum definition not allowed here')
        if ctx.overridable:
            if s.systring != 'enum':
                error(pos, 'C struct/union cannot be declared cpdef')
        return p_struct_enum(s, pos, ctx)
    elif s.sy == 'IDENT' and s.systring == 'fused':
        return p_fused_definition(s, pos, ctx)
    else:
        return p_c_func_or_var_declaration(s, pos, ctx)