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
@lower_constant(FunctionType)
def lower_constant_function_type(context, builder, typ, pyval):
    typ = typ.get_precise()
    if isinstance(pyval, CFunc):
        addr = pyval._wrapper_address
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.addr = context.add_dynamic_addr(builder, addr, info=str(typ))
        sfunc.pyaddr = context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)
        return sfunc._getvalue()
    if isinstance(pyval, Dispatcher):
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.pyaddr = context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)
        return sfunc._getvalue()
    if isinstance(pyval, WrapperAddressProtocol):
        addr = pyval.__wrapper_address__()
        assert typ.check_signature(pyval.signature())
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.addr = context.add_dynamic_addr(builder, addr, info=str(typ))
        sfunc.pyaddr = context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)
        return sfunc._getvalue()
    raise NotImplementedError('lower_constant_struct_function_type({}, {}, {}, {})'.format(context, builder, typ, pyval))