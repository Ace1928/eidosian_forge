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
def buf_lookup_c_code(proto, defin, name, nd):
    """
    Similar to strided lookup, but can assume that the last dimension
    doesn't need a multiplication as long as.
    Still we keep the same signature for now.
    """
    if nd == 1:
        proto.putln('#define %s(type, buf, i0, s0) ((type)buf + i0)' % name)
    else:
        args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
        offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(nd - 1)])
        proto.putln('#define %s(type, buf, %s) ((type)((char*)buf + %s) + i%d)' % (name, args, offset, nd - 1))