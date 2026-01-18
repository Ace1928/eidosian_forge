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
def p_include_statement(s, ctx):
    pos = s.position()
    s.next()
    unicode_include_file_name = p_string_literal(s, 'u')[2]
    s.expect_newline('Syntax error in include statement')
    if s.compile_time_eval:
        include_file_name = unicode_include_file_name
        include_file_path = s.context.find_include_file(include_file_name, pos)
        if include_file_path:
            s.included_files.append(include_file_name)
            with Utils.open_source_file(include_file_path) as f:
                source_desc = FileSourceDescriptor(include_file_path)
                s2 = PyrexScanner(f, source_desc, s, source_encoding=f.encoding, parse_comments=s.parse_comments)
                tree = p_statement_list(s2, ctx)
            return tree
        else:
            return None
    else:
        return Nodes.PassStatNode(pos)