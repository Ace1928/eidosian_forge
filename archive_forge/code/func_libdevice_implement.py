from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs
def libdevice_implement(func, retty, nbargs):

    def core(context, builder, sig, args):
        lmod = builder.module
        fretty = context.get_value_type(retty)
        fargtys = [context.get_value_type(arg.ty) for arg in nbargs]
        fnty = ir.FunctionType(fretty, fargtys)
        fn = cgutils.get_or_insert_function(lmod, fnty, func)
        return builder.call(fn, args)
    key = getattr(libdevice, func[5:])
    argtys = [arg.ty for arg in args if not arg.is_ptr]
    lower(key, *argtys)(core)