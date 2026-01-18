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
def get_suboffset_dim():
    if not suboffset_dim_temp:
        suboffset_dim = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = -1;' % suboffset_dim)
        suboffset_dim_temp.append(suboffset_dim)
    return suboffset_dim_temp[0]