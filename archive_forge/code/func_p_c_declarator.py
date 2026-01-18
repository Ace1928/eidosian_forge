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
def p_c_declarator(s, ctx=Ctx(), empty=0, is_type=0, cmethod_flag=0, assignable=0, nonempty=0, calling_convention_allowed=0):
    pos = s.position()
    if s.sy == '(':
        s.next()
        if s.sy == ')' or looking_at_name(s):
            base = Nodes.CNameDeclaratorNode(pos, name=s.context.intern_ustring(u''), cname=None)
            result = p_c_func_declarator(s, pos, ctx, base, cmethod_flag)
        else:
            result = p_c_declarator(s, ctx, empty=empty, is_type=is_type, cmethod_flag=cmethod_flag, nonempty=nonempty, calling_convention_allowed=1)
            s.expect(')')
    else:
        result = p_c_simple_declarator(s, ctx, empty, is_type, cmethod_flag, assignable, nonempty)
    if not calling_convention_allowed and result.calling_convention and (s.sy != '('):
        error(s.position(), '%s on something that is not a function' % result.calling_convention)
    while s.sy in ('[', '('):
        pos = s.position()
        if s.sy == '[':
            result = p_c_array_declarator(s, result)
        else:
            s.next()
            result = p_c_func_declarator(s, pos, ctx, result, cmethod_flag)
        cmethod_flag = 0
    return result