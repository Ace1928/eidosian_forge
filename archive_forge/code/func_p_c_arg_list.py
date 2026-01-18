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
def p_c_arg_list(s, ctx=Ctx(), in_pyfunc=0, cmethod_flag=0, nonempty_declarators=0, kw_only=0, annotated=1):
    args = []
    is_self_arg = cmethod_flag
    while s.sy not in c_arg_list_terminators:
        args.append(p_c_arg_decl(s, ctx, in_pyfunc, is_self_arg, nonempty=nonempty_declarators, kw_only=kw_only, annotated=annotated))
        if s.sy != ',':
            break
        s.next()
        is_self_arg = 0
    return args