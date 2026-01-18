import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
    arg_index = 0 if argidx is None else argidx
    m = self.module
    fnty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
    decl = cgutils.get_or_insert_function(m, fnty, 'llvm.dbg.declare')
    mdtype = self._var_type(lltype, size, datamodel=datamodel)
    name = name.replace('.', '$')
    mdlocalvar = m.add_debug_info('DILocalVariable', {'name': name, 'arg': arg_index, 'scope': self.subprograms[-1], 'file': self.difile, 'line': line, 'type': mdtype})
    mdexpr = m.add_debug_info('DIExpression', {})
    return builder.call(decl, [allocavalue, mdlocalvar, mdexpr])