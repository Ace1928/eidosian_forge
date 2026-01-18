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
def put_acquire_arg_buffer(entry, code, pos):
    buffer_aux = entry.buffer_aux
    getbuffer = get_getbuffer_call(code, entry.cname, buffer_aux, entry.type)
    code.putln('{')
    code.putln('__Pyx_BufFmt_StackElem __pyx_stack[%d];' % entry.type.dtype.struct_nesting_depth())
    code.putln(code.error_goto_if('%s == -1' % getbuffer, pos))
    code.putln('}')
    put_unpack_buffer_aux_into_scope(entry, code)