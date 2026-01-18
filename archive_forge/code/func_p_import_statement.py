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
def p_import_statement(s):
    pos = s.position()
    kind = s.sy
    s.next()
    items = [p_dotted_name(s, as_allowed=1)]
    while s.sy == ',':
        s.next()
        items.append(p_dotted_name(s, as_allowed=1))
    stats = []
    is_absolute = Future.absolute_import in s.context.future_directives
    for pos, target_name, dotted_name, as_name in items:
        if kind == 'cimport':
            stat = Nodes.CImportStatNode(pos, module_name=dotted_name, as_name=as_name, is_absolute=is_absolute)
        else:
            stat = Nodes.SingleAssignmentNode(pos, lhs=ExprNodes.NameNode(pos, name=as_name or target_name), rhs=ExprNodes.ImportNode(pos, module_name=ExprNodes.IdentifierStringNode(pos, value=dotted_name), level=0 if is_absolute else None, get_top_level_module='.' in dotted_name and as_name is None, name_list=None))
        stats.append(stat)
    return Nodes.StatListNode(pos, stats=stats)