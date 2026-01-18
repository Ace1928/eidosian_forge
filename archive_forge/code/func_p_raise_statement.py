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
def p_raise_statement(s):
    pos = s.position()
    s.next()
    exc_type = None
    exc_value = None
    exc_tb = None
    cause = None
    if s.sy not in statement_terminators:
        exc_type = p_test(s)
        if s.sy == ',':
            s.next()
            exc_value = p_test(s)
            if s.sy == ',':
                s.next()
                exc_tb = p_test(s)
        elif s.sy == 'from':
            s.next()
            cause = p_test(s)
    if exc_type or exc_value or exc_tb:
        return Nodes.RaiseStatNode(pos, exc_type=exc_type, exc_value=exc_value, exc_tb=exc_tb, cause=cause)
    else:
        return Nodes.ReraiseStatNode(pos)