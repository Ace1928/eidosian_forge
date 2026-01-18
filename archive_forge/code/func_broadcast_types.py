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
def broadcast_types(src, dst):
    n = abs(src.ndim - dst.ndim)
    if src.ndim < dst.ndim:
        return (insert_newaxes(src, n), dst)
    else:
        return (src, insert_newaxes(dst, n))