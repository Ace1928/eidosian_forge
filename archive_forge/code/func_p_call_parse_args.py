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
def p_call_parse_args(s, allow_genexp=True):
    pos = s.position()
    s.next()
    positional_args = []
    keyword_args = []
    starstar_seen = False
    last_was_tuple_unpack = False
    while s.sy != ')':
        if s.sy == '*':
            if starstar_seen:
                s.error('Non-keyword arg following keyword arg', pos=s.position())
            s.next()
            positional_args.append(p_test(s))
            last_was_tuple_unpack = True
        elif s.sy == '**':
            s.next()
            keyword_args.append(p_test(s))
            starstar_seen = True
        else:
            arg = p_namedexpr_test(s)
            if s.sy == '=':
                s.next()
                if not arg.is_name:
                    s.error("Expected an identifier before '='", pos=arg.pos)
                encoded_name = s.context.intern_ustring(arg.name)
                keyword = ExprNodes.IdentifierStringNode(arg.pos, value=encoded_name)
                arg = p_test(s)
                keyword_args.append((keyword, arg))
            else:
                if keyword_args:
                    s.error('Non-keyword arg following keyword arg', pos=arg.pos)
                if positional_args and (not last_was_tuple_unpack):
                    positional_args[-1].append(arg)
                else:
                    positional_args.append([arg])
                last_was_tuple_unpack = False
        if s.sy != ',':
            break
        s.next()
    if s.sy in ('for', 'async'):
        if not keyword_args and (not last_was_tuple_unpack):
            if len(positional_args) == 1 and len(positional_args[0]) == 1:
                positional_args = [[p_genexp(s, positional_args[0][0])]]
    s.expect(')')
    return (positional_args or [[]], keyword_args)