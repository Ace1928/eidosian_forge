from numba.core.extending import intrinsic
from llvmlite import ir
from numba.core import types, cgutils
@intrinsic
def grab_byte(typingctx, data, offset):

    def impl(context, builder, signature, args):
        data, idx = args
        ptr = builder.bitcast(data, ir.IntType(8).as_pointer())
        ch = builder.load(builder.gep(ptr, [idx]))
        return ch
    sig = types.uint8(types.voidptr, types.intp)
    return (sig, impl)