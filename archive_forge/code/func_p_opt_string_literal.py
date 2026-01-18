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
def p_opt_string_literal(s, required_type='u'):
    if s.sy != 'BEGIN_STRING':
        return None
    pos = s.position()
    kind, bytes_value, unicode_value = p_string_literal(s, required_type)
    if required_type == 'u':
        if kind == 'f':
            s.error('f-string not allowed here', pos)
        return unicode_value
    elif required_type == 'b':
        return bytes_value
    else:
        s.error('internal parser configuration error')