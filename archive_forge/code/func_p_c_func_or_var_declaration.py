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
def p_c_func_or_var_declaration(s, pos, ctx):
    cmethod_flag = ctx.level in ('c_class', 'c_class_pxd')
    modifiers = p_c_modifiers(s)
    base_type = p_c_base_type(s, nonempty=1, templates=ctx.templates)
    declarator = p_c_declarator(s, ctx(modifiers=modifiers), cmethod_flag=cmethod_flag, assignable=1, nonempty=1)
    declarator.overridable = ctx.overridable
    if s.sy == 'IDENT' and s.systring == 'const' and (ctx.level == 'cpp_class'):
        s.next()
        is_const_method = 1
    else:
        is_const_method = 0
    if s.sy == '->':
        s.error('Return type annotation is not allowed in cdef/cpdef signatures. Please define it before the function name, as in C signatures.', fatal=False)
        s.next()
        p_test(s)
    if s.sy == ':':
        if ctx.level not in ('module', 'c_class', 'module_pxd', 'c_class_pxd', 'cpp_class') and (not ctx.templates):
            s.error('C function definition not allowed here')
        doc, suite = p_suite_with_docstring(s, Ctx(level='function'))
        result = Nodes.CFuncDefNode(pos, visibility=ctx.visibility, base_type=base_type, declarator=declarator, body=suite, doc=doc, modifiers=modifiers, api=ctx.api, overridable=ctx.overridable, is_const_method=is_const_method)
    else:
        if is_const_method:
            declarator.is_const_method = is_const_method
        declarators = [declarator]
        while s.sy == ',':
            s.next()
            if s.sy == 'NEWLINE':
                break
            declarator = p_c_declarator(s, ctx, cmethod_flag=cmethod_flag, assignable=1, nonempty=1)
            declarators.append(declarator)
        doc_line = s.start_line + 1
        s.expect_newline('Syntax error in C variable declaration', ignore_semicolon=True)
        if ctx.level in ('c_class', 'c_class_pxd') and s.start_line == doc_line:
            doc = p_doc_string(s)
        else:
            doc = None
        result = Nodes.CVarDefNode(pos, visibility=ctx.visibility, base_type=base_type, declarators=declarators, in_pxd=ctx.level in ('module_pxd', 'c_class_pxd'), doc=doc, api=ctx.api, modifiers=modifiers, overridable=ctx.overridable)
    return result