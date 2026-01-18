from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def load_memview_cy_utility(util_code_name, context=None, **kwargs):
    return CythonUtilityCode.load(util_code_name, 'MemoryView.pyx', context=context, **kwargs)