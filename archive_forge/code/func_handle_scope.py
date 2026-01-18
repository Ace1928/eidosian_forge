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
def handle_scope(self, node, scope):
    scope_items = scope.entries.items()
    bufvars = [entry for name, entry in scope_items if entry.type.is_buffer]
    if len(bufvars) > 0:
        bufvars.sort(key=lambda entry: entry.name)
        self.buffers_exists = True
    memviewslicevars = [entry for name, entry in scope_items if entry.type.is_memoryviewslice]
    if len(memviewslicevars) > 0:
        self.buffers_exists = True
    for name, entry in scope_items:
        if name == 'memoryview' and isinstance(entry.utility_code_definition, CythonUtilityCode):
            self.using_memoryview = True
            break
    del scope_items
    if isinstance(node, ModuleNode) and len(bufvars) > 0:
        raise CompileError(node.pos, 'Buffer vars not allowed in module scope')
    for entry in bufvars:
        if entry.type.dtype.is_ptr:
            raise CompileError(node.pos, 'Buffers with pointer types not yet supported.')
        name = entry.name
        buftype = entry.type
        if buftype.ndim > Options.buffer_max_dims:
            raise CompileError(node.pos, 'Buffer ndims exceeds Options.buffer_max_dims = %d' % Options.buffer_max_dims)
        if buftype.ndim > self.max_ndim:
            self.max_ndim = buftype.ndim

        def decvar(type, prefix):
            cname = scope.mangle(prefix, name)
            aux_var = scope.declare_var(name=None, cname=cname, type=type, pos=node.pos)
            if entry.is_arg:
                aux_var.used = True
            return aux_var
        auxvars = ((PyrexTypes.c_pyx_buffer_nd_type, Naming.pybuffernd_prefix), (PyrexTypes.c_pyx_buffer_type, Naming.pybufferstruct_prefix))
        pybuffernd, rcbuffer = [decvar(type, prefix) for type, prefix in auxvars]
        entry.buffer_aux = Symtab.BufferAux(pybuffernd, rcbuffer)
    scope.buffer_entries = bufvars
    self.scope = scope