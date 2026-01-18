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
@property
def aux_dict(self):
    """Get dictionary representation of auxiliary states arrays.

        Returns
        -------
        aux_dict : dict of str to NDArray
            The dictionary that maps name of auxiliary states to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the auxiliary states.
        """
    if self._aux_dict is None:
        self._aux_dict = Executor._get_dict(self._symbol.list_auxiliary_states(), self.aux_arrays)
    return self._aux_dict