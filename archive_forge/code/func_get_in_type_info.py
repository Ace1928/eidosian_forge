from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def get_in_type_info(self):
    definitions = []
    for n, arg in enumerate(self.node.args):
        type_const = _get_type_constant(self.node.pos, arg.type)
        definitions.append(_ArgumentInfo(arg.type, type_const))
    return definitions