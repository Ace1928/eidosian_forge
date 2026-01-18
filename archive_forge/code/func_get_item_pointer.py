import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def get_item_pointer(context, builder, aryty, ary, inds, wraparound=False, boundscheck=False):
    shapes = unpack_tuple(builder, ary.shape, count=aryty.ndim)
    strides = unpack_tuple(builder, ary.strides, count=aryty.ndim)
    return get_item_pointer2(context, builder, data=ary.data, shape=shapes, strides=strides, layout=aryty.layout, inds=inds, wraparound=wraparound, boundscheck=boundscheck)