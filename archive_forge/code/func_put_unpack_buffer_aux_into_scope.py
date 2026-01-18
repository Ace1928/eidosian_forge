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
def put_unpack_buffer_aux_into_scope(buf_entry, code):
    buffer_aux, mode = (buf_entry.buffer_aux, buf_entry.type.mode)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    fldnames = ['strides', 'shape']
    if mode == 'full':
        fldnames.append('suboffsets')
    ln = []
    for i in range(buf_entry.type.ndim):
        for fldname in fldnames:
            ln.append('%s.diminfo[%d].%s = %s.rcbuffer->pybuffer.%s[%d];' % (pybuffernd_struct, i, fldname, pybuffernd_struct, fldname, i))
    code.putln(' '.join(ln))