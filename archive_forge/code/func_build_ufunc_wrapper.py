from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def build_ufunc_wrapper(library, context, fname, signature, objmode, cres):
    """
    Wrap the scalar function with a loop that iterates over the arguments

    Returns
    -------
    (library, env, name)
    """
    assert isinstance(fname, str)
    byte_t = ir.IntType(8)
    byte_ptr_t = ir.PointerType(byte_t)
    byte_ptr_ptr_t = ir.PointerType(byte_ptr_t)
    intp_t = context.get_value_type(types.intp)
    intp_ptr_t = ir.PointerType(intp_t)
    fnty = ir.FunctionType(ir.VoidType(), [byte_ptr_ptr_t, intp_ptr_t, intp_ptr_t, byte_ptr_t])
    wrapperlib = context.codegen().create_library('ufunc_wrapper')
    wrapper_module = wrapperlib.create_ir_module('')
    if objmode:
        func_type = context.call_conv.get_function_type(types.pyobject, [types.pyobject] * len(signature.args))
    else:
        func_type = context.call_conv.get_function_type(signature.return_type, signature.args)
    func = ir.Function(wrapper_module, func_type, name=fname)
    func.attributes.add('alwaysinline')
    wrapper = ir.Function(wrapper_module, fnty, '__ufunc__.' + func.name)
    arg_args, arg_dims, arg_steps, arg_data = wrapper.args
    arg_args.name = 'args'
    arg_dims.name = 'dims'
    arg_steps.name = 'steps'
    arg_data.name = 'data'
    builder = IRBuilder(wrapper.append_basic_block('entry'))
    envname = context.get_env_name(cres.fndesc)
    env = cres.environment
    envptr = builder.load(context.declare_env_global(builder.module, envname))
    loopcount = builder.load(arg_dims, name='loopcount')
    arrays = []
    for i, typ in enumerate(signature.args):
        arrays.append(UArrayArg(context, builder, arg_args, arg_steps, i, typ))
    out = UArrayArg(context, builder, arg_args, arg_steps, len(arrays), signature.return_type)
    offsets = []
    zero = context.get_constant(types.intp, 0)
    for _ in arrays:
        p = cgutils.alloca_once(builder, intp_t)
        offsets.append(p)
        builder.store(zero, p)
    store_offset = cgutils.alloca_once(builder, intp_t)
    builder.store(zero, store_offset)
    unit_strided = cgutils.true_bit
    for ary in arrays:
        unit_strided = builder.and_(unit_strided, ary.is_unit_strided)
    pyapi = context.get_python_api(builder)
    if objmode:
        gil = pyapi.gil_ensure()
        with cgutils.for_range(builder, loopcount, intp=intp_t):
            build_obj_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, envptr, env)
        pyapi.gil_release(gil)
        builder.ret_void()
    else:
        with builder.if_else(unit_strided) as (is_unit_strided, is_strided):
            with is_unit_strided:
                with cgutils.for_range(builder, loopcount, intp=intp_t) as loop:
                    build_fast_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, loop.index, pyapi, env=envptr)
            with is_strided:
                with cgutils.for_range(builder, loopcount, intp=intp_t):
                    build_slow_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env=envptr)
        builder.ret_void()
    del builder
    wrapperlib.add_ir_module(wrapper_module)
    wrapperlib.add_linking_library(library)
    return _wrapper_info(library=wrapperlib, env=env, name=wrapper.name)