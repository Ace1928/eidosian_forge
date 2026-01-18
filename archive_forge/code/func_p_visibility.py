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
def p_visibility(s, prev_visibility):
    pos = s.position()
    visibility = prev_visibility
    if s.sy == 'IDENT' and s.systring in ('extern', 'public', 'readonly'):
        visibility = s.systring
        if prev_visibility != 'private' and visibility != prev_visibility:
            s.error("Conflicting visibility options '%s' and '%s'" % (prev_visibility, visibility), fatal=False)
        s.next()
    return visibility