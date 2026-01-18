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
def p_c_enum_line(s, ctx, items):
    if s.sy != 'pass':
        p_c_enum_item(s, ctx, items)
        while s.sy == ',':
            s.next()
            if s.sy in ('NEWLINE', 'EOF'):
                break
            p_c_enum_item(s, ctx, items)
    else:
        s.next()
    s.expect_newline('Syntax error in enum item list')