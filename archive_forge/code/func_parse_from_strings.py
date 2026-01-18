from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
def parse_from_strings(name, code, pxds=None, level=None, initial_pos=None, context=None, allow_struct_enum_decorator=False, in_utility_code=False):
    """
    Utility method to parse a (unicode) string of code. This is mostly
    used for internal Cython compiler purposes (creating code snippets
    that transforms should emit, as well as unit testing).

    code - a unicode string containing Cython (module-level) code
    name - a descriptive name for the code source (to use in error messages etc.)
    in_utility_code - used to suppress some messages from utility code. False by default
                      because some generated code snippets like properties and dataclasses
                      probably want to see those messages.

    RETURNS

    The tree, i.e. a ModuleNode. The ModuleNode's scope attribute is
    set to the scope used when parsing.
    """
    if context is None:
        context = StringParseContext(name)
    assert isinstance(code, _unicode), 'unicode code snippets only please'
    encoding = 'UTF-8'
    module_name = name
    if initial_pos is None:
        initial_pos = (name, 1, 0)
    code_source = StringSourceDescriptor(name, code)
    if in_utility_code:
        code_source.in_utility_code = True
    scope = context.find_module(module_name, pos=initial_pos, need_pxd=False)
    buf = StringIO(code)
    scanner = PyrexScanner(buf, code_source, source_encoding=encoding, scope=scope, context=context, initial_pos=initial_pos)
    ctx = Parsing.Ctx(allow_struct_enum_decorator=allow_struct_enum_decorator)
    if level is None:
        tree = Parsing.p_module(scanner, 0, module_name, ctx=ctx)
        tree.scope = scope
        tree.is_pxd = False
    else:
        tree = Parsing.p_code(scanner, level=level, ctx=ctx)
    tree.scope = scope
    return tree