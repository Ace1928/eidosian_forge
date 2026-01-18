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
def p_f_string(s, unicode_value, pos, is_raw):
    values = []
    next_start = 0
    size = len(unicode_value)
    builder = StringEncoding.UnicodeLiteralBuilder()
    _parse_seq = _parse_escape_sequences_raw if is_raw else _parse_escape_sequences
    while next_start < size:
        end = next_start
        match = _parse_seq(unicode_value, next_start)
        if match is None:
            error(_f_string_error_pos(pos, unicode_value, next_start), 'Invalid escape sequence')
        next_start = match.end()
        part = match.group()
        c = part[0]
        if c == '\\':
            if not is_raw and len(part) > 1:
                _append_escape_sequence('f', builder, part, s)
            else:
                builder.append(part)
        elif c == '{':
            if part == '{{':
                builder.append('{')
            else:
                if builder.chars:
                    values.append(ExprNodes.UnicodeNode(pos, value=builder.getstring()))
                    builder = StringEncoding.UnicodeLiteralBuilder()
                next_start, expr_nodes = p_f_string_expr(s, unicode_value, pos, next_start, is_raw)
                values.extend(expr_nodes)
        elif c == '}':
            if part == '}}':
                builder.append('}')
            else:
                error(_f_string_error_pos(pos, unicode_value, end), "f-string: single '}' is not allowed")
        else:
            builder.append(part)
    if builder.chars:
        values.append(ExprNodes.UnicodeNode(pos, value=builder.getstring()))
    return values