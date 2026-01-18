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
def p_cpp_class_definition(s, pos, ctx):
    s.next()
    class_name = p_ident(s)
    cname = p_opt_cname(s)
    if cname is None and ctx.namespace is not None:
        cname = ctx.namespace + '::' + class_name
    if s.sy == '.':
        error(pos, 'Qualified class name not allowed C++ class')
    if s.sy == '[':
        s.next()
        templates = [p_template_definition(s)]
        while s.sy == ',':
            s.next()
            templates.append(p_template_definition(s))
        s.expect(']')
        template_names = [name for name, required in templates]
    else:
        templates = None
        template_names = None
    if s.sy == '(':
        s.next()
        base_classes = [p_c_base_type(s, templates=template_names)]
        while s.sy == ',':
            s.next()
            base_classes.append(p_c_base_type(s, templates=template_names))
        s.expect(')')
    else:
        base_classes = []
    if s.sy == '[':
        error(s.position(), 'Name options not allowed for C++ class')
    nogil = p_nogil(s)
    if s.sy == ':':
        s.next()
        s.expect('NEWLINE')
        s.expect_indent()
        p_doc_string(s)
        attributes = []
        body_ctx = Ctx(visibility=ctx.visibility, level='cpp_class', nogil=nogil or ctx.nogil)
        body_ctx.templates = template_names
        while s.sy != 'DEDENT':
            if s.sy != 'pass':
                attributes.append(p_cpp_class_attribute(s, body_ctx))
            else:
                s.next()
                s.expect_newline('Expected a newline')
        s.expect_dedent()
    else:
        attributes = None
        s.expect_newline('Syntax error in C++ class definition')
    return Nodes.CppClassNode(pos, name=class_name, cname=cname, base_classes=base_classes, visibility=ctx.visibility, in_pxd=ctx.level == 'module_pxd', attributes=attributes, templates=templates)