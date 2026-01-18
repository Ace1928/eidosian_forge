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
def p_print_statement(s):
    pos = s.position()
    ends_with_comma = 0
    s.next()
    if s.sy == '>>':
        s.next()
        stream = p_test(s)
        if s.sy == ',':
            s.next()
            ends_with_comma = s.sy in ('NEWLINE', 'EOF')
    else:
        stream = None
    args = []
    if s.sy not in ('NEWLINE', 'EOF'):
        args.append(p_test(s))
        while s.sy == ',':
            s.next()
            if s.sy in ('NEWLINE', 'EOF'):
                ends_with_comma = 1
                break
            args.append(p_test(s))
    arg_tuple = ExprNodes.TupleNode(pos, args=args)
    return Nodes.PrintStatNode(pos, arg_tuple=arg_tuple, stream=stream, append_newline=not ends_with_comma)