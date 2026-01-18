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
def p_c_simple_declarator(s, ctx, empty, is_type, cmethod_flag, assignable, nonempty):
    pos = s.position()
    calling_convention = p_calling_convention(s)
    if s.sy in ('*', '**'):
        is_ptrptr = s.sy == '**'
        s.next()
        const_pos = s.position()
        is_const = s.systring == 'const' and s.sy == 'IDENT'
        if is_const:
            s.next()
        base = p_c_declarator(s, ctx, empty=empty, is_type=is_type, cmethod_flag=cmethod_flag, assignable=assignable, nonempty=nonempty)
        if is_const:
            base = Nodes.CConstDeclaratorNode(const_pos, base=base)
        if is_ptrptr:
            base = Nodes.CPtrDeclaratorNode(pos, base=base)
        result = Nodes.CPtrDeclaratorNode(pos, base=base)
    elif s.sy == '&' or (s.sy == '&&' and s.context.cpp):
        node_class = Nodes.CppRvalueReferenceDeclaratorNode if s.sy == '&&' else Nodes.CReferenceDeclaratorNode
        s.next()
        base = p_c_declarator(s, ctx, empty=empty, is_type=is_type, cmethod_flag=cmethod_flag, assignable=assignable, nonempty=nonempty)
        result = node_class(pos, base=base)
    else:
        rhs = None
        if s.sy == 'IDENT':
            name = s.systring
            if empty:
                error(s.position(), 'Declarator should be empty')
            s.next()
            cname = p_opt_cname(s)
            if name != 'operator' and s.sy == '=' and assignable:
                s.next()
                rhs = p_test(s)
        else:
            if nonempty:
                error(s.position(), 'Empty declarator')
            name = ''
            cname = None
        if cname is None and ctx.namespace is not None and nonempty:
            cname = ctx.namespace + '::' + name
        if name == 'operator' and ctx.visibility == 'extern' and nonempty:
            op = s.sy
            if [1 for c in op if c in '+-*/<=>!%&|([^~,']:
                s.next()
                if op == '(':
                    s.expect(')')
                    op = '()'
                elif op == '[':
                    s.expect(']')
                    op = '[]'
                elif op in ('-', '+', '|', '&') and s.sy == op:
                    op *= 2
                    s.next()
                elif s.sy == '=':
                    op += s.sy
                    s.next()
                if op not in supported_overloaded_operators:
                    s.error("Overloading operator '%s' not yet supported." % op, fatal=False)
                name += op
            elif op == 'IDENT':
                op = s.systring
                if op not in supported_overloaded_operators:
                    s.error("Overloading operator '%s' not yet supported." % op, fatal=False)
                name = name + ' ' + op
                s.next()
        result = Nodes.CNameDeclaratorNode(pos, name=name, cname=cname, default=rhs)
    result.calling_convention = calling_convention
    return result