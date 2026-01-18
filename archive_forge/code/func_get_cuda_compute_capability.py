import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def get_cuda_compute_capability(ctx):
    """Returns the cuda compute capability of the input `ctx`.

    Parameters
    ----------
    ctx : Context
        GPU context whose corresponding cuda compute capability is to be retrieved.

    Returns
    -------
    cuda_compute_capability : int
        CUDA compute capability. For example, it returns 70 for CUDA arch equal to `sm_70`.

    References
    ----------
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549#file-cuda_check-py
    """
    if ctx.device_type != 'gpu':
        raise ValueError('Expecting a gpu context to get cuda compute capability, while received ctx {}'.format(str(ctx)))
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError('could not load any of: ' + ' '.join(libnames))
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()
    ret = cuda.cuInit(0)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuInit failed with erro code {}: {}'.format(ret, error_str.value.decode()))
    ret = cuda.cuDeviceGet(ctypes.byref(device), ctx.device_id)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuDeviceGet failed with error code {}: {}'.format(ret, error_str.value.decode()))
    ret = cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuDeviceComputeCapability failed with error code {}: {}'.format(ret, error_str.value.decode()))
    return cc_major.value * 10 + cc_minor.value