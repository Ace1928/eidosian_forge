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
def p_sign_and_longness(s):
    signed = 1
    longness = 0
    while s.sy == 'IDENT' and s.systring in sign_and_longness_words:
        if s.systring == 'unsigned':
            signed = 0
        elif s.systring == 'signed':
            signed = 2
        elif s.systring == 'short':
            longness = -1
        elif s.systring == 'long':
            longness += 1
        s.next()
    return (signed, longness)