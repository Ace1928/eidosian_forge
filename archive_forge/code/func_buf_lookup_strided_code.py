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
def buf_lookup_strided_code(proto, defin, name, nd):
    """
    Generates a buffer lookup function for the right number
    of dimensions. The function gives back a void* at the right location.
    """
    args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
    offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(nd)])
    proto.putln('#define %s(type, buf, %s) (type)((char*)buf + %s)' % (name, args, offset))