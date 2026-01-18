import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def populate_array(array, data, shape, strides, itemsize, meminfo, parent=None):
    """
    Helper function for populating array structures.
    This avoids forgetting to set fields.

    *shape* and *strides* can be Python tuples or LLVM arrays.
    """
    context = array._context
    builder = array._builder
    datamodel = array._datamodel
    standard_array = types.Array(types.float64, 1, 'C')
    standard_array_type_datamodel = context.data_model_manager[standard_array]
    required_fields = set(standard_array_type_datamodel._fields)
    datamodel_fields = set(datamodel._fields)
    if required_fields & datamodel_fields != required_fields:
        missing = required_fields - datamodel_fields
        msg = f'The datamodel for type {array._fe_type} is missing field{('s' if len(missing) > 1 else '')} {missing}.'
        raise ValueError(msg)
    if meminfo is None:
        meminfo = Constant(context.get_value_type(datamodel.get_type('meminfo')), None)
    intp_t = context.get_value_type(types.intp)
    if isinstance(shape, (tuple, list)):
        shape = cgutils.pack_array(builder, shape, intp_t)
    if isinstance(strides, (tuple, list)):
        strides = cgutils.pack_array(builder, strides, intp_t)
    if isinstance(itemsize, int):
        itemsize = intp_t(itemsize)
    attrs = dict(shape=shape, strides=strides, data=data, itemsize=itemsize, meminfo=meminfo)
    if parent is None:
        attrs['parent'] = Constant(context.get_value_type(datamodel.get_type('parent')), None)
    else:
        attrs['parent'] = parent
    nitems = context.get_constant(types.intp, 1)
    unpacked_shape = cgutils.unpack_tuple(builder, shape, shape.type.count)
    for axlen in unpacked_shape:
        nitems = builder.mul(nitems, axlen, flags=['nsw'])
    attrs['nitems'] = nitems
    got_fields = set(attrs.keys())
    if got_fields != required_fields:
        raise ValueError('missing {0}'.format(required_fields - got_fields))
    for k, v in attrs.items():
        setattr(array, k, v)
    return array