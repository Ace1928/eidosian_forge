from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
@box(FunctionType)
def box_function_type(typ, val, c):
    typ = typ.get_precise()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    pyaddr_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    raw_ptr = c.builder.inttoptr(sfunc.pyaddr, c.pyapi.pyobj)
    with c.builder.if_then(cgutils.is_null(c.builder, raw_ptr), likely=False):
        cstr = f'first-class function {typ} parent object not set'
        c.pyapi.err_set_string('PyExc_MemoryError', cstr)
        c.builder.ret(c.pyapi.get_null_object())
    c.builder.store(raw_ptr, pyaddr_ptr)
    cfunc = c.builder.load(pyaddr_ptr)
    c.pyapi.incref(cfunc)
    return cfunc