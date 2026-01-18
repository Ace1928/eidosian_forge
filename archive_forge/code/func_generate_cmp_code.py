from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
def generate_cmp_code(code, op, funcname, node, fields):
    if node.scope.lookup_here(funcname):
        return
    names = [name for name, field in fields.items() if field.compare.value and (not field.is_initvar)]
    code.add_code_lines(['def %s(self, other):' % funcname, '    if other.__class__ is not self.__class__:        return NotImplemented', '    cdef %s other_cast' % node.class_name, '    other_cast = <%s>other' % node.class_name])
    checks = []
    op_without_equals = op.replace('=', '')
    for name in names:
        if op != '==':
            code.add_code_line('    if self.%s %s other_cast.%s: return True' % (name, op_without_equals, name))
        code.add_code_line('    if self.%s != other_cast.%s: return False' % (name, name))
    if '=' in op:
        code.add_code_line('    return True')
    else:
        code.add_code_line('    return False')