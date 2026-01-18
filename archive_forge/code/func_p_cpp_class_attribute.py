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
def p_cpp_class_attribute(s, ctx):
    decorators = None
    if s.sy == '@':
        decorators = p_decorators(s)
    if s.systring == 'cppclass':
        return p_cpp_class_definition(s, s.position(), ctx)
    elif s.systring == 'ctypedef':
        return p_ctypedef_statement(s, ctx)
    elif s.sy == 'IDENT' and s.systring in struct_enum_union:
        if s.systring != 'enum':
            return p_cpp_class_definition(s, s.position(), ctx)
        else:
            return p_struct_enum(s, s.position(), ctx)
    else:
        node = p_c_func_or_var_declaration(s, s.position(), ctx)
        if decorators is not None:
            tup = (Nodes.CFuncDefNode, Nodes.CVarDefNode, Nodes.CClassDefNode)
            if ctx.allow_struct_enum_decorator:
                tup += (Nodes.CStructOrUnionDefNode, Nodes.CEnumDefNode)
            if not isinstance(node, tup):
                s.error('Decorators can only be followed by functions or classes')
            node.decorators = decorators
        return node