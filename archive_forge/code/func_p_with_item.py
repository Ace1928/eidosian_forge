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
def p_with_item(s, is_async):
    pos = s.position()
    if not s.in_python_file and s.sy == 'IDENT' and (s.systring in ('nogil', 'gil')):
        if is_async:
            s.error('with gil/nogil cannot be async')
        state = s.systring
        s.next()
        condition = None
        if s.sy == '(':
            s.next()
            condition = p_test(s)
            s.expect(')')
        return (Nodes.GILStatNode, pos, {'state': state, 'condition': condition})
    else:
        manager = p_test(s)
        target = None
        if s.sy == 'IDENT' and s.systring == 'as':
            s.next()
            target = p_starred_expr(s)
        return (Nodes.WithStatNode, pos, {'manager': manager, 'target': target, 'is_async': is_async})