from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def from_native_generator(self, val, typ, env=None):
    """
        Make a Numba generator (a _dynfunc.Generator instance) from a
        generator structure pointer *val*.
        *env* is an optional _dynfunc.Environment instance to be wrapped
        in the generator.
        """
    llty = self.context.get_data_type(typ)
    assert not llty.is_pointer
    gen_struct_size = self.context.get_abi_sizeof(llty)
    gendesc = self.context.get_generator_desc(typ)
    genfnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    genfn = self._get_function(genfnty, name=gendesc.llvm_cpython_wrapper_name)
    finalizerty = ir.FunctionType(ir.VoidType(), [self.voidptr])
    if typ.has_finalizer:
        finalizer = self._get_function(finalizerty, name=gendesc.llvm_finalizer_name)
    else:
        finalizer = Constant(ir.PointerType(finalizerty), None)
    fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t, self.voidptr, ir.PointerType(genfnty), ir.PointerType(finalizerty), self.voidptr])
    fn = self._get_function(fnty, name='numba_make_generator')
    state_size = Constant(self.py_ssize_t, gen_struct_size)
    initial_state = self.builder.bitcast(val, self.voidptr)
    if env is None:
        env = self.get_null_object()
    env = self.builder.bitcast(env, self.voidptr)
    return self.builder.call(fn, (state_size, initial_state, genfn, finalizer, env))