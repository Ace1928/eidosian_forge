from numba.core.extending import intrinsic
from llvmlite import ir
from numba.core import types, cgutils
@intrinsic
def memcpy_region(typingctx, dst, dst_offset, src, src_offset, nbytes, align):
    """Copy nbytes from *(src + src_offset) to *(dst + dst_offset)"""

    def codegen(context, builder, signature, args):
        [dst_val, dst_offset_val, src_val, src_offset_val, nbytes_val, align_val] = args
        src_ptr = builder.gep(src_val, [src_offset_val])
        dst_ptr = builder.gep(dst_val, [dst_offset_val])
        cgutils.raw_memcpy(builder, dst_ptr, src_ptr, nbytes_val, align_val)
        return context.get_dummy_value()
    sig = types.void(types.voidptr, types.intp, types.voidptr, types.intp, types.intp, types.intp)
    return (sig, codegen)