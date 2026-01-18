import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
def from_cuda_array_interface(desc, owner=None, sync=True):
    """Create a DeviceNDArray from a cuda-array-interface description.
    The ``owner`` is the owner of the underlying memory.
    The resulting DeviceNDArray will acquire a reference from it.

    If ``sync`` is ``True``, then the imported stream (if present) will be
    synchronized.
    """
    version = desc.get('version')
    if 1 <= version:
        mask = desc.get('mask')
        if mask is not None:
            raise NotImplementedError('Masked arrays are not supported')
    shape = desc['shape']
    strides = desc.get('strides')
    dtype = np.dtype(desc['typestr'])
    shape, strides, dtype = prepare_shape_strides_dtype(shape, strides, dtype, order='C')
    size = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    devptr = driver.get_devptr_for_active_ctx(desc['data'][0])
    data = driver.MemoryPointer(current_context(), devptr, size=size, owner=owner)
    stream_ptr = desc.get('stream', None)
    if stream_ptr is not None:
        stream = external_stream(stream_ptr)
        if sync and config.CUDA_ARRAY_INTERFACE_SYNC:
            stream.synchronize()
    else:
        stream = 0
    da = devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=data, stream=stream)
    return da