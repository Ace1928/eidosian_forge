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
def p_code(s, level=None, ctx=Ctx):
    body = p_statement_list(s, ctx(level=level), first_statement=1)
    if s.sy != 'EOF':
        s.error('Syntax error in statement [%s,%s]' % (repr(s.sy), repr(s.systring)))
    return body