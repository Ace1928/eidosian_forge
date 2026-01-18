from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
@property
def provide_data(self):
    """The name and shape of data provided by this iterator."""
    batch_axis = self.layout.find('N')
    return [DataDesc(k, tuple(list(v.shape[:batch_axis]) + [self.batch_size] + list(v.shape[batch_axis + 1:])), v.dtype, layout=self.layout) for k, v in self.data]