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
@staticmethod
def get_batch_axis(layout):
    """Get the dimension that corresponds to the batch size.

        When data parallelism is used, the data will be automatically split and
        concatenated along the batch-size dimension. Axis can be -1, which means
        the whole array will be copied for each data-parallelism device.

        Parameters
        ----------
        layout : str
            layout string. For example, "NCHW".

        Returns
        -------
        int
            An axis indicating the batch_size dimension.
        """
    if layout is None:
        return 0
    return layout.find('N')