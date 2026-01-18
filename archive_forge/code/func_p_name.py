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
def p_name(s, name):
    pos = s.position()
    if not s.compile_time_expr and name in s.compile_time_env:
        value = s.compile_time_env.lookup_here(name)
        node = wrap_compile_time_constant(pos, value)
        if node is not None:
            return node
    return ExprNodes.NameNode(pos, name=name)