from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def get_out_type_info(self):
    if self.node.return_type.is_ctuple:
        components = self.node.return_type.components
    else:
        components = [self.node.return_type]
    definitions = []
    for n, type in enumerate(components):
        definitions.append(_ArgumentInfo(type, _get_type_constant(self.node.pos, type)))
    return definitions