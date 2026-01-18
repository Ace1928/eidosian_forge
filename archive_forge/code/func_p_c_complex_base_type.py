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
def p_c_complex_base_type(s, templates=None):
    pos = s.position()
    s.next()
    base_type = p_c_base_type(s, templates=templates)
    declarator = p_c_declarator(s, empty=True)
    type_node = Nodes.CComplexBaseTypeNode(pos, base_type=base_type, declarator=declarator)
    if s.sy == ',':
        components = [type_node]
        while s.sy == ',':
            s.next()
            if s.sy == ')':
                break
            base_type = p_c_base_type(s, templates=templates)
            declarator = p_c_declarator(s, empty=True)
            components.append(Nodes.CComplexBaseTypeNode(pos, base_type=base_type, declarator=declarator))
        type_node = Nodes.CTupleBaseTypeNode(pos, components=components)
    s.expect(')')
    if s.sy == '[':
        if is_memoryviewslice_access(s):
            type_node = p_memoryviewslice_access(s, type_node)
        else:
            type_node = p_buffer_or_template(s, type_node, templates)
    return type_node