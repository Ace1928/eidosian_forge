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
def p_c_class_definition(s, pos, ctx):
    s.next()
    module_path = []
    class_name = p_ident(s)
    while s.sy == '.':
        s.next()
        module_path.append(class_name)
        class_name = p_ident(s)
    if module_path and ctx.visibility != 'extern':
        error(pos, "Qualified class name only allowed for 'extern' C class")
    if module_path and s.sy == 'IDENT' and (s.systring == 'as'):
        s.next()
        as_name = p_ident(s)
    else:
        as_name = class_name
    objstruct_name = None
    typeobj_name = None
    bases = None
    check_size = None
    if s.sy == '(':
        positional_args, keyword_args = p_call_parse_args(s, allow_genexp=False)
        if keyword_args:
            s.error('C classes cannot take keyword bases.')
        bases, _ = p_call_build_packed_args(pos, positional_args, keyword_args)
    if bases is None:
        bases = ExprNodes.TupleNode(pos, args=[])
    if s.sy == '[':
        if ctx.visibility not in ('public', 'extern') and (not ctx.api):
            error(s.position(), "Name options only allowed for 'public', 'api', or 'extern' C class")
        objstruct_name, typeobj_name, check_size = p_c_class_options(s)
    if s.sy == ':':
        if ctx.level == 'module_pxd':
            body_level = 'c_class_pxd'
        else:
            body_level = 'c_class'
        doc, body = p_suite_with_docstring(s, Ctx(level=body_level))
    else:
        s.expect_newline('Syntax error in C class definition')
        doc = None
        body = None
    if ctx.visibility == 'extern':
        if not module_path:
            error(pos, "Module name required for 'extern' C class")
        if typeobj_name:
            error(pos, "Type object name specification not allowed for 'extern' C class")
    elif ctx.visibility == 'public':
        if not objstruct_name:
            error(pos, "Object struct name specification required for 'public' C class")
        if not typeobj_name:
            error(pos, "Type object name specification required for 'public' C class")
    elif ctx.visibility == 'private':
        if ctx.api:
            if not objstruct_name:
                error(pos, "Object struct name specification required for 'api' C class")
            if not typeobj_name:
                error(pos, "Type object name specification required for 'api' C class")
    else:
        error(pos, "Invalid class visibility '%s'" % ctx.visibility)
    return Nodes.CClassDefNode(pos, visibility=ctx.visibility, typedef_flag=ctx.typedef_flag, api=ctx.api, module_name='.'.join(module_path), class_name=class_name, as_name=as_name, bases=bases, objstruct_name=objstruct_name, typeobj_name=typeobj_name, check_size=check_size, in_pxd=ctx.level == 'module_pxd', doc=doc, body=body)