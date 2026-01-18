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
def p_positional_and_keyword_args(s, end_sy_set, templates=None):
    """
    Parses positional and keyword arguments. end_sy_set
    should contain any s.sy that terminate the argument list.
    Argument expansion (* and **) are not allowed.

    Returns: (positional_args, keyword_args)
    """
    positional_args = []
    keyword_args = []
    pos_idx = 0
    while s.sy not in end_sy_set:
        if s.sy == '*' or s.sy == '**':
            s.error('Argument expansion not allowed here.', fatal=False)
        parsed_type = False
        if s.sy == 'IDENT' and s.peek()[0] == '=':
            ident = s.systring
            s.next()
            s.next()
            if looking_at_expr(s):
                arg = p_test(s)
            else:
                base_type = p_c_base_type(s, templates=templates)
                declarator = p_c_declarator(s, empty=1)
                arg = Nodes.CComplexBaseTypeNode(base_type.pos, base_type=base_type, declarator=declarator)
                parsed_type = True
            keyword_node = ExprNodes.IdentifierStringNode(arg.pos, value=ident)
            keyword_args.append((keyword_node, arg))
            was_keyword = True
        else:
            if looking_at_expr(s):
                arg = p_test(s)
            else:
                base_type = p_c_base_type(s, templates=templates)
                declarator = p_c_declarator(s, empty=1)
                arg = Nodes.CComplexBaseTypeNode(base_type.pos, base_type=base_type, declarator=declarator)
                parsed_type = True
            positional_args.append(arg)
            pos_idx += 1
            if len(keyword_args) > 0:
                s.error('Non-keyword arg following keyword arg', pos=arg.pos)
        if s.sy != ',':
            if s.sy not in end_sy_set:
                if parsed_type:
                    s.error('Unmatched %s' % ' or '.join(end_sy_set))
            break
        s.next()
    return (positional_args, keyword_args)