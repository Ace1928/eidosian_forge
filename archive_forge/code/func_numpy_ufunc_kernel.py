import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):
    arguments = [_prepare_argument(context, builder, arg, tyarg) for arg, tyarg in zip(args, sig.args)]
    if len(arguments) < ufunc.nin:
        raise RuntimeError('Not enough inputs to {}, expected {} got {}'.format(ufunc.__name__, ufunc.nin, len(arguments)))
    for out_i, ret_ty in enumerate(_unpack_output_types(ufunc, sig)):
        if ufunc.nin + out_i >= len(arguments):
            if isinstance(ret_ty, types.ArrayCompatible):
                output = _build_array(context, builder, ret_ty, sig.args, arguments)
            else:
                output = _prepare_argument(context, builder, ir.Constant(context.get_value_type(ret_ty), None), ret_ty)
            arguments.append(output)
        elif context.enable_nrt:
            context.nrt.incref(builder, ret_ty, args[ufunc.nin + out_i])
    inputs = arguments[:ufunc.nin]
    outputs = arguments[ufunc.nin:]
    assert len(outputs) == ufunc.nout
    outer_sig = _ufunc_loop_sig([a.base_type for a in outputs], [a.base_type for a in inputs])
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)
    indices = [inp.create_iter_indices() for inp in inputs]
    loopshape = outputs[0].shape
    input_layouts = [inp.layout for inp in inputs if isinstance(inp, _ArrayHelper)]
    num_c_layout = len([x for x in input_layouts if x == 'C'])
    num_f_layout = len([x for x in input_layouts if x == 'F'])
    if num_f_layout > num_c_layout:
        order = 'F'
    else:
        order = 'C'
    with cgutils.loop_nest(builder, loopshape, intp=intpty, order=order) as loop_indices:
        vals_in = []
        for i, (index, arg) in enumerate(zip(indices, inputs)):
            index.update_indices(loop_indices, i)
            vals_in.append(arg.load_data(index.as_values()))
        vals_out = _unpack_output_values(ufunc, builder, kernel.generate(*vals_in))
        for val_out, output in zip(vals_out, outputs):
            output.store_data(loop_indices, val_out)
    out = _pack_output_values(ufunc, context, builder, sig.return_type, [o.return_val for o in outputs])
    return impl_ret_new_ref(context, builder, sig.return_type, out)