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
def p_index(s, base):
    pos = s.position()
    s.next()
    subscripts, is_single_value = p_subscript_list(s)
    if is_single_value and len(subscripts[0]) == 2:
        start, stop = subscripts[0]
        result = ExprNodes.SliceIndexNode(pos, base=base, start=start, stop=stop)
    else:
        indexes = make_slice_nodes(pos, subscripts)
        if is_single_value:
            index = indexes[0]
        else:
            index = ExprNodes.TupleNode(pos, args=indexes)
        result = ExprNodes.IndexNode(pos, base=base, index=index)
    s.expect(']')
    return result