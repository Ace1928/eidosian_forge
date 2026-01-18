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
def p_ignorable_statement(s):
    """
    Parses any kind of ignorable statement that is allowed in .pxd files.
    """
    if s.sy == 'BEGIN_STRING':
        pos = s.position()
        string_node = p_atom(s)
        s.expect_newline('Syntax error in string', ignore_semicolon=True)
        return Nodes.ExprStatNode(pos, expr=string_node)
    return None