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
def p_varargslist(s, terminator=')', annotated=1):
    args = p_c_arg_list(s, in_pyfunc=1, nonempty_declarators=1, annotated=annotated)
    star_arg = None
    starstar_arg = None
    if s.sy == '/':
        if len(args) == 0:
            s.error("Got zero positional-only arguments despite presence of positional-only specifier '/'")
        s.next()
        for arg in args:
            arg.pos_only = 1
        if s.sy == ',':
            s.next()
            args.extend(p_c_arg_list(s, in_pyfunc=1, nonempty_declarators=1, annotated=annotated))
        elif s.sy != terminator:
            s.error('Syntax error in Python function argument list')
    if s.sy == '*':
        s.next()
        if s.sy == 'IDENT':
            star_arg = p_py_arg_decl(s, annotated=annotated)
        if s.sy == ',':
            s.next()
            args.extend(p_c_arg_list(s, in_pyfunc=1, nonempty_declarators=1, kw_only=1, annotated=annotated))
        elif s.sy != terminator:
            s.error('Syntax error in Python function argument list')
    if s.sy == '**':
        s.next()
        starstar_arg = p_py_arg_decl(s, annotated=annotated)
    if s.sy == ',':
        s.next()
    return (args, star_arg, starstar_arg)