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
def p_c_simple_base_type(s, nonempty, templates=None):
    is_basic = 0
    signed = 1
    longness = 0
    complex = 0
    module_path = []
    pos = s.position()
    is_const = is_volatile = 0
    while s.sy == 'IDENT':
        if s.systring == 'const':
            if is_const:
                error(pos, "Duplicate 'const'")
            is_const = 1
        elif s.systring == 'volatile':
            if is_volatile:
                error(pos, "Duplicate 'volatile'")
            is_volatile = 1
        else:
            break
        s.next()
    if is_const or is_volatile:
        base_type = p_c_base_type(s, nonempty=nonempty, templates=templates)
        if isinstance(base_type, Nodes.MemoryViewSliceTypeNode):
            base_type.base_type_node = Nodes.CConstOrVolatileTypeNode(pos, base_type=base_type.base_type_node, is_const=is_const, is_volatile=is_volatile)
            return base_type
        return Nodes.CConstOrVolatileTypeNode(pos, base_type=base_type, is_const=is_const, is_volatile=is_volatile)
    if s.sy != 'IDENT':
        error(pos, "Expected an identifier, found '%s'" % s.sy)
    if looking_at_base_type(s):
        is_basic = 1
        if s.sy == 'IDENT' and s.systring in special_basic_c_types:
            signed, longness = special_basic_c_types[s.systring]
            name = s.systring
            s.next()
        else:
            signed, longness = p_sign_and_longness(s)
            if s.sy == 'IDENT' and s.systring in basic_c_type_names:
                name = s.systring
                s.next()
            else:
                name = 'int'
        if s.sy == 'IDENT' and s.systring == 'complex':
            complex = 1
            s.next()
    elif looking_at_dotted_name(s):
        name = s.systring
        s.next()
        while s.sy == '.':
            module_path.append(name)
            s.next()
            name = p_ident(s)
    else:
        name = s.systring
        name_pos = s.position()
        s.next()
        if nonempty and s.sy != 'IDENT':
            if s.sy == '(':
                old_pos = s.position()
                s.next()
                if s.sy == '*' or s.sy == '**' or s.sy == '&' or (s.sy == 'IDENT' and s.systring in calling_convention_words):
                    s.put_back(u'(', u'(', old_pos)
                else:
                    s.put_back(u'(', u'(', old_pos)
                    s.put_back(u'IDENT', name, name_pos)
                    name = None
            elif s.sy not in ('*', '**', '[', '&'):
                s.put_back(u'IDENT', name, name_pos)
                name = None
    type_node = Nodes.CSimpleBaseTypeNode(pos, name=name, module_path=module_path, is_basic_c_type=is_basic, signed=signed, complex=complex, longness=longness, templates=templates)
    if s.sy == '[':
        if is_memoryviewslice_access(s):
            type_node = p_memoryviewslice_access(s, type_node)
        else:
            type_node = p_buffer_or_template(s, type_node, templates)
    if s.sy == '.':
        s.next()
        name = p_ident(s)
        type_node = Nodes.CNestedBaseTypeNode(pos, base_type=type_node, name=name)
    return type_node