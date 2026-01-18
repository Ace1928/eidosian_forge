import ctypes
import sys
from .base import _LIB
from .base import c_array, c_str_array, c_handle_array, py_str, build_param_doc as _build_param_doc
from .base import mx_uint, mx_float, FunctionHandle
from .base import check_call
from .ndarray import NDArray, _new_empty_handle
def generic_torch_function(*args, **kwargs):
    """Invoke this function by passing in parameters.

        Parameters
        ----------
        *args
            Positional arguments of inputs (both scalar and `NDArray`).

        Returns
        -------
        out : NDArray
            The result NDArray(tuple) of result of computation.
        """
    ndargs = []
    arg_format = ''
    value = ''
    for arg in args:
        if isinstance(arg, NDArray):
            ndargs.append(arg)
            arg_format += 'n'
            value += ','
        elif isinstance(arg, int):
            arg_format += 'i'
            value += str(arg) + ','
        elif isinstance(arg, str):
            arg_format += 's'
            value += str(arg) + ','
        elif isinstance(arg, float):
            arg_format += 'f'
            value += str(arg) + ','
        elif isinstance(arg, bool):
            arg_format += 'b'
            value += str(arg) + ','
    value = value[:-1]
    if len(ndargs) == n_used_vars:
        ndargs = [NDArray(_new_empty_handle()) for _ in range(n_mutate_vars)] + ndargs
        arg_format = 'n' * n_mutate_vars + arg_format
        value = ',' * n_mutate_vars + value
    elif len(ndargs) == n_mutate_vars + n_used_vars:
        pass
    else:
        raise AssertionError(('Incorrect number of input NDArrays. ' + 'Need to be either %d (inputs) or %d ' + '(output buffer) + %d (input)') % (n_used_vars, n_mutate_vars, n_used_vars))
    kwargs['format'] = arg_format
    kwargs['args'] = value
    for k in kwargs:
        kwargs[k] = str(kwargs[k])
    check_call(_LIB.MXFuncInvokeEx(handle, c_handle_array(ndargs[n_mutate_vars:]), c_array(mx_float, []), c_handle_array(ndargs[:n_mutate_vars]), ctypes.c_int(len(kwargs)), c_str_array(kwargs.keys()), c_str_array(kwargs.values())))
    if n_mutate_vars == 1:
        return ndargs[0]
    else:
        return ndargs[:n_mutate_vars]