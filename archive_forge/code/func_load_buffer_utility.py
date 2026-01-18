from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def load_buffer_utility(util_code_name, context=None, **kwargs):
    if context is None:
        return UtilityCode.load(util_code_name, 'Buffer.c', **kwargs)
    else:
        return TempitaUtilityCode.load(util_code_name, 'Buffer.c', context=context, **kwargs)