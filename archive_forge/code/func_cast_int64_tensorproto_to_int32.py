import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def cast_int64_tensorproto_to_int32(initializer: onnx.TensorProto, cast: bool=False):
    """
    Casts in place the input TensorProto data to int32. Its data is assumed to be of type int64,
    and in case some values are out of range, they are cast to the min/max representable
    value in int32.
    """
    original_name = initializer.name
    array = np.copy(numpy_helper.to_array(initializer))
    if not array.dtype == np.int64:
        raise TypeError('Expecting a `TensorProto` of type `int64` (represented as `7` in onnx.TensorProto) in the function int64_tensorproto_to_int32, but got {array.dtype}.')
    array[array > np.iinfo(np.int32).max] = np.iinfo(np.int32).max
    array[array < np.iinfo(np.int32).min] = np.iinfo(np.int32).min
    if cast:
        array = array.astype(np.int32)
    array.setflags(write=0)
    tensor = numpy_helper.from_array(array)
    initializer.CopyFrom(tensor)
    initializer.name = original_name