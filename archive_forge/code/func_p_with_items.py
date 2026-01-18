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
def p_with_items(s, is_async=False):
    """
    Copied from CPython:
    | 'with' '(' a[asdl_withitem_seq*]=','.with_item+ ','? ')' ':' b=block {
        _PyAST_With(a, b, NULL, EXTRA) }
    | 'with' a[asdl_withitem_seq*]=','.with_item+ ':' tc=[TYPE_COMMENT] b=block {
        _PyAST_With(a, b, NEW_TYPE_COMMENT(p, tc), EXTRA) }
    Therefore the first thing to try is the bracket-enclosed
    version and if that fails try the regular version
    """
    brackets_succeeded = False
    items = ()
    if s.sy == '(':
        with tentatively_scan(s) as errors:
            s.next()
            items = p_with_items_list(s, is_async)
            s.expect(')')
            if s.sy != ':':
                s.error('')
        brackets_succeeded = not errors
    if not brackets_succeeded:
        items = p_with_items_list(s, is_async)
    body = p_suite(s)
    for cls, pos, kwds in reversed(items):
        body = cls(pos, body=body, **kwds)
    return body