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
def put_release_buffer_code(code, entry):
    code.globalstate.use_utility_code(acquire_utility_code)
    code.putln('__Pyx_SafeReleaseBuffer(&%s.rcbuffer->pybuffer);' % entry.buffer_aux.buflocal_nd_var.cname)