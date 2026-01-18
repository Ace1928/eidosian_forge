import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib
def compile_multi3(context):
    """
    Compile the multi3() helper function used by LLVM
    for 128-bit multiplication on 32-bit platforms.
    """
    codegen = context.codegen()
    library = codegen.create_library('multi3')
    ir_mod = library.create_ir_module('multi3')
    i64 = ir.IntType(64)
    i128 = ir.IntType(128)
    lower_mask = ir.Constant(i64, 4294967295)
    _32 = ir.Constant(i64, 32)
    _64 = ir.Constant(i128, 64)
    fn_type = ir.FunctionType(i128, [i128, i128])
    fn = ir.Function(ir_mod, fn_type, name='multi3')
    a, b = fn.args
    bb = fn.append_basic_block()
    builder = ir.IRBuilder(bb)
    al = builder.trunc(a, i64)
    bl = builder.trunc(b, i64)
    ah = builder.trunc(builder.ashr(a, _64), i64)
    bh = builder.trunc(builder.ashr(b, _64), i64)
    rl = builder.mul(builder.and_(al, lower_mask), builder.and_(bl, lower_mask))
    t = builder.lshr(rl, _32)
    rl = builder.and_(rl, lower_mask)
    t = builder.add(t, builder.mul(builder.lshr(al, _32), builder.and_(bl, lower_mask)))
    rl = builder.add(rl, builder.shl(t, _32))
    rh = builder.lshr(t, _32)
    t = builder.lshr(rl, _32)
    rl = builder.and_(rl, lower_mask)
    t = builder.add(t, builder.mul(builder.lshr(bl, _32), builder.and_(al, lower_mask)))
    rl = builder.add(rl, builder.shl(t, _32))
    rh = builder.add(rh, builder.lshr(t, _32))
    rh = builder.add(rh, builder.mul(builder.lshr(al, _32), builder.lshr(bl, _32)))
    rh = builder.add(rh, builder.mul(bh, al))
    rh = builder.add(rh, builder.mul(bl, ah))
    r = builder.zext(rl, i128)
    r = builder.add(r, builder.shl(builder.zext(rh, i128), _64))
    builder.ret(r)
    library.add_ir_module(ir_mod)
    library.finalize()
    return library