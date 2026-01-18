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
def p_cdef_extern_block(s, pos, ctx):
    if ctx.overridable:
        error(pos, 'cdef extern blocks cannot be declared cpdef')
    include_file = None
    s.expect('from')
    if s.sy == '*':
        s.next()
    else:
        include_file = p_string_literal(s, 'u')[2]
    ctx = ctx(cdef_flag=1, visibility='extern')
    if s.systring == 'namespace':
        s.next()
        ctx.namespace = p_string_literal(s, 'u')[2]
    if p_nogil(s):
        ctx.nogil = 1
    verbatim_include, body = p_suite_with_docstring(s, ctx, True)
    return Nodes.CDefExternNode(pos, include_file=include_file, verbatim_include=verbatim_include, body=body, namespace=ctx.namespace)