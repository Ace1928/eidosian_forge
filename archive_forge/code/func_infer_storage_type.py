import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
def infer_storage_type(self, in_stype):
    """infer_storage_type interface. Used to infer storage type of
        inputs and outputs in the forward pass. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        in_stype : list of stypes, valid stypes are default, row_sparse and
            csr

        Returns
        -------
        in_stype : list
            list of argument stypes.
        out_stype : list
            list of output types calculated from in_stype,
            in the same order as declared in list_outputs.
        aux_type : Optional, list
            list of aux types calculated from in_stype,
            in the same order as declared in list_auxiliary_states.
        """
    for i, stype in enumerate(in_stype):
        assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], "Default infer_storage_type implementation doesnt allow non default stypes: found non default stype '%s' for in_stype[%d]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default input/output stypes" % (stype, i)
    return (in_stype, [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]] * len(self.list_outputs()), [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]] * len(self.list_auxiliary_states()))