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
def load_dispatcher(self, fnty, argtypes):
    builder = self.pyapi.builder
    tyctx = self.pyapi.context
    m = builder.module
    gv = ir.GlobalVariable(m, self.pyapi.pyobj, name=m.get_unique_name('cached_objmode_dispatcher'))
    gv.initializer = gv.type.pointee(None)
    gv.linkage = 'internal'
    bb_end = builder.append_basic_block('bb_end')
    if serialize.is_serialiable(fnty.dispatcher):
        serialized_dispatcher = self.pyapi.serialize_object((fnty.dispatcher, tuple(argtypes)))
        compile_args = self.pyapi.unserialize(serialized_dispatcher)
        failed_unser = cgutils.is_null(builder, compile_args)
        with builder.if_then(failed_unser):
            builder.branch(bb_end)
    cached = builder.load(gv)
    with builder.if_then(cgutils.is_null(builder, cached)):
        if serialize.is_serialiable(fnty.dispatcher):
            cls = type(self)
            compiler = self.pyapi.unserialize(self.pyapi.serialize_object(cls._call_objmode_dispatcher))
            callee = self.pyapi.call_function_objargs(compiler, [compile_args])
            self.pyapi.decref(compiler)
            self.pyapi.decref(compile_args)
        else:
            entry_pt = fnty.dispatcher.compile(tuple(argtypes))
            callee = tyctx.add_dynamic_addr(builder, id(entry_pt), info='with_objectmode')
        self.pyapi.incref(callee)
        builder.store(callee, gv)
    builder.branch(bb_end)
    builder.position_at_end(bb_end)
    callee = builder.load(gv)
    return callee