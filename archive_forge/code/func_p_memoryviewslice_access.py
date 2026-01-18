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
def p_memoryviewslice_access(s, base_type_node):
    pos = s.position()
    s.next()
    subscripts, _ = p_subscript_list(s)
    for subscript in subscripts:
        if len(subscript) < 2:
            s.error("An axis specification in memoryview declaration does not have a ':'.")
    s.expect(']')
    indexes = make_slice_nodes(pos, subscripts)
    result = Nodes.MemoryViewSliceTypeNode(pos, base_type_node=base_type_node, axes=indexes)
    return result