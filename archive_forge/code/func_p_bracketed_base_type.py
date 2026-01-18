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
def p_bracketed_base_type(s, base_type_node, nonempty, empty):
    if empty and (not nonempty):
        return base_type_node
    elif not empty and nonempty:
        if is_memoryviewslice_access(s):
            return p_memoryviewslice_access(s, base_type_node)
        else:
            return p_buffer_or_template(s, base_type_node, None)
    elif not empty and (not nonempty):
        if is_memoryviewslice_access(s):
            return p_memoryviewslice_access(s, base_type_node)
        else:
            return base_type_node