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
def copy_params_from(self, arg_params, aux_params=None, allow_extra_params=False):
    """Copy parameters from arg_params, aux_params into executor's internal array.

        Parameters
        ----------
        arg_params : dict of str to NDArray
            Parameters, dict of name to NDArray of arguments.

        aux_params : dict of str to NDArray, optional
            Parameters, dict of name to NDArray of auxiliary states.

        allow_extra_params : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Raises
        ------
        ValueError
            If there is additional parameters in the dict but ``allow_extra_params=False``.

        Examples
        --------
        >>> # set parameters with existing model checkpoint
        >>> model_prefix = 'mx_mlp'
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
        >>> texec.copy_params_from(arg_params, aux_params)
        """
    for name, array in arg_params.items():
        if name in self.arg_dict:
            dst = self.arg_dict[name]
            if dst.dtype == np.dtype([('bfloat16', np.uint16)]):
                cast_array = ndarray.amp_cast(array, dtype=dst.dtype)
                cast_array.copyto(dst)
            else:
                array.astype(dst.dtype).copyto(dst)
        elif not allow_extra_params:
            raise ValueError('Find name "%s" that is not in the arguments' % name)
    if aux_params is None:
        return
    for name, array in aux_params.items():
        if name in self.aux_dict:
            dst = self.aux_dict[name]
            if dst.dtype == np.dtype([('bfloat16', np.uint16)]):
                cast_array = ndarray.amp_cast(array, dtype=dst.dtype)
                cast_array.copyto(dst)
            else:
                array.astype(dst.dtype).copyto(dst)
        elif not allow_extra_params:
            raise ValueError('Find name %s that is not in the auxiliary states' % name)