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
def p_buffer_or_template(s, base_type_node, templates):
    pos = s.position()
    s.next()
    positional_args, keyword_args = p_positional_and_keyword_args(s, (']',), templates)
    s.expect(']')
    if s.sy == '[':
        base_type_node = p_buffer_or_template(s, base_type_node, templates)
    keyword_dict = ExprNodes.DictNode(pos, key_value_pairs=[ExprNodes.DictItemNode(pos=key.pos, key=key, value=value) for key, value in keyword_args])
    result = Nodes.TemplatedTypeNode(pos, positional_args=positional_args, keyword_args=keyword_dict, base_type_node=base_type_node)
    return result