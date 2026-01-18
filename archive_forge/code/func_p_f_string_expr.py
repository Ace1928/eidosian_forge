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
def p_f_string_expr(s, unicode_value, pos, starting_index, is_raw):
    i = starting_index
    size = len(unicode_value)
    conversion_char = terminal_char = format_spec = None
    format_spec_str = None
    expr_text = None
    NO_CHAR = 2 ** 30
    nested_depth = 0
    quote_char = NO_CHAR
    in_triple_quotes = False
    backslash_reported = False
    while True:
        if i >= size:
            break
        c = unicode_value[i]
        if quote_char != NO_CHAR:
            if c == '\\':
                if not backslash_reported:
                    error(_f_string_error_pos(pos, unicode_value, i), 'backslashes not allowed in f-strings')
                backslash_reported = True
            elif c == quote_char:
                if in_triple_quotes:
                    if i + 2 < size and unicode_value[i + 1] == c and (unicode_value[i + 2] == c):
                        in_triple_quotes = False
                        quote_char = NO_CHAR
                        i += 2
                else:
                    quote_char = NO_CHAR
        elif c in '\'"':
            quote_char = c
            if i + 2 < size and unicode_value[i + 1] == c and (unicode_value[i + 2] == c):
                in_triple_quotes = True
                i += 2
        elif c in '{[(':
            nested_depth += 1
        elif nested_depth != 0 and c in '}])':
            nested_depth -= 1
        elif c == '#':
            error(_f_string_error_pos(pos, unicode_value, i), 'format string cannot include #')
        elif nested_depth == 0 and c in '><=!:}':
            if i + 1 < size and c in '!=><':
                if unicode_value[i + 1] == '=':
                    i += 2
                    continue
                elif c in '><':
                    i += 1
                    continue
            terminal_char = c
            break
        i += 1
    expr_str = unicode_value[starting_index:i].replace('\r\n', '\n').replace('\r', '\n')
    expr_pos = (pos[0], pos[1], pos[2] + starting_index + 2)
    if not expr_str.strip():
        error(_f_string_error_pos(pos, unicode_value, starting_index), 'empty expression not allowed in f-string')
    if terminal_char == '=':
        i += 1
        while i < size and unicode_value[i].isspace():
            i += 1
        if i < size:
            terminal_char = unicode_value[i]
            expr_text = unicode_value[starting_index:i]
    if terminal_char == '!':
        i += 1
        if i + 2 > size:
            pass
        else:
            conversion_char = unicode_value[i]
            i += 1
            terminal_char = unicode_value[i]
    if terminal_char == ':':
        in_triple_quotes = False
        in_string = False
        nested_depth = 0
        start_format_spec = i + 1
        while True:
            if i >= size:
                break
            c = unicode_value[i]
            if not in_triple_quotes and (not in_string):
                if c == '{':
                    nested_depth += 1
                elif c == '}':
                    if nested_depth > 0:
                        nested_depth -= 1
                    else:
                        terminal_char = c
                        break
            if c in '\'"':
                if not in_string and i + 2 < size and (unicode_value[i + 1] == c) and (unicode_value[i + 2] == c):
                    in_triple_quotes = not in_triple_quotes
                    i += 2
                elif not in_triple_quotes:
                    in_string = not in_string
            i += 1
        format_spec_str = unicode_value[start_format_spec:i]
    if expr_text and conversion_char is None and (format_spec_str is None):
        conversion_char = 'r'
    if terminal_char != '}':
        error(_f_string_error_pos(pos, unicode_value, i), "missing '}' in format string expression" + (", found '%s'" % terminal_char if terminal_char else ''))
    buf = StringIO('(%s)' % expr_str)
    scanner = PyrexScanner(buf, expr_pos[0], parent_scanner=s, source_encoding=s.source_encoding, initial_pos=expr_pos)
    expr = p_testlist(scanner)
    if conversion_char is not None and (not ExprNodes.FormattedValueNode.find_conversion_func(conversion_char)):
        error(expr_pos, "invalid conversion character '%s'" % conversion_char)
    if format_spec_str:
        format_spec = ExprNodes.JoinedStrNode(pos, values=p_f_string(s, format_spec_str, pos, is_raw))
    nodes = []
    if expr_text:
        nodes.append(ExprNodes.UnicodeNode(pos, value=StringEncoding.EncodedString(expr_text)))
    nodes.append(ExprNodes.FormattedValueNode(pos, value=expr, conversion_char=conversion_char, format_spec=format_spec))
    return (i + 1, nodes)