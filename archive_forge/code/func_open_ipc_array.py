import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@contextlib.contextmanager
@require_context
def open_ipc_array(handle, shape, dtype, strides=None, offset=0):
    """
    A context manager that opens a IPC *handle* (*CUipcMemHandle*) that is
    represented as a sequence of bytes (e.g. *bytes*, tuple of int)
    and represent it as an array of the given *shape*, *strides* and *dtype*.
    The *strides* can be omitted.  In that case, it is assumed to be a 1D
    C contiguous array.

    Yields a device array.

    The IPC handle is closed automatically when context manager exits.
    """
    dtype = np.dtype(dtype)
    size = np.prod(shape) * dtype.itemsize
    if driver.USE_NV_BINDING:
        driver_handle = driver.binding.CUipcMemHandle()
        driver_handle.reserved = handle
    else:
        driver_handle = driver.drvapi.cu_ipc_mem_handle(*handle)
    ipchandle = driver.IpcHandle(None, driver_handle, size, offset=offset)
    yield ipchandle.open_array(current_context(), shape=shape, strides=strides, dtype=dtype)
    ipchandle.close()