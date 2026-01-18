from array import array as py_array
import ctypes
import copy
import numpy as np
from .base import _LIB
from .base import mx_uint, NDArrayHandle, SymbolHandle, ExecutorHandle, py_str, mx_int
from .base import check_call, c_handle_array, c_array_buf, c_str_array
from . import ndarray
from .ndarray import NDArray
from .ndarray import _ndarray_cls
from .executor_manager import _split_input_slice, _check_arguments, _load_data, _load_label
def set_monitor_callback(self, callback, monitor_all=False):
    """Install callback for monitor.

        Parameters
        ----------
        callback : function
            Takes a string and an NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input and output, otherwise monitor output only.

        Examples
        --------
        >>> def mon_callback(*args, **kwargs):
        >>>     print("Do your stuff here.")
        >>>
        >>> texe.set_monitor_callback(mon_callback)
        """
    cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
    self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
    self._monitor_all = monitor_all
    check_call(_LIB.MXExecutorSetMonitorCallbackEX(self.handle, self._monitor_callback, None, ctypes.c_int(monitor_all)))