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
def p_for_bounds(s, allow_testlist=True, is_async=False):
    target = p_for_target(s)
    if s.sy == 'in':
        s.next()
        iterator = p_for_iterator(s, allow_testlist, is_async=is_async)
        return dict(target=target, iterator=iterator)
    elif not s.in_python_file and (not is_async):
        if s.sy == 'from':
            s.next()
            bound1 = p_bit_expr(s)
        else:
            bound1, target = (target, None)
        rel1 = p_for_from_relation(s)
        name2_pos = s.position()
        name2 = p_ident(s)
        rel2_pos = s.position()
        rel2 = p_for_from_relation(s)
        bound2 = p_bit_expr(s)
        step = p_for_from_step(s)
        if target is None:
            target = ExprNodes.NameNode(name2_pos, name=name2)
        elif not target.is_name:
            error(target.pos, 'Target of for-from statement must be a variable name')
        elif name2 != target.name:
            error(name2_pos, 'Variable name in for-from range does not match target')
        if rel1[0] != rel2[0]:
            error(rel2_pos, 'Relation directions in for-from do not match')
        return dict(target=target, bound1=bound1, relation1=rel1, relation2=rel2, bound2=bound2, step=step)
    else:
        s.expect('in')
        return {}