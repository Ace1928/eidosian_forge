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
def p_class_statement(s, decorators):
    pos = s.position()
    s.next()
    class_name = EncodedString(p_ident(s))
    class_name.encoding = s.source_encoding
    arg_tuple = None
    keyword_dict = None
    if s.sy == '(':
        positional_args, keyword_args = p_call_parse_args(s, allow_genexp=False)
        arg_tuple, keyword_dict = p_call_build_packed_args(pos, positional_args, keyword_args)
    if arg_tuple is None:
        arg_tuple = ExprNodes.TupleNode(pos, args=[])
    doc, body = p_suite_with_docstring(s, Ctx(level='class'))
    return Nodes.PyClassDefNode(pos, name=class_name, bases=arg_tuple, keyword_args=keyword_dict, doc=doc, body=body, decorators=decorators, force_py3_semantics=s.context.language_level >= 3)