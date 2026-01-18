from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
def fake_arg(arg):
    _, arg = functools.reduce(lambda ty_val, extension: extension.prepare_args(*ty_val, stream=0, retr=retr), self.extensions, (None, arg))
    if isinstance(arg, np.ndarray) and arg.ndim > 0:
        ret = wrap_arg(arg).to_device(retr)
    elif isinstance(arg, ArgHint):
        ret = arg.to_device(retr)
    elif isinstance(arg, np.void):
        ret = FakeCUDAArray(arg)
    else:
        ret = arg
    if isinstance(ret, FakeCUDAArray):
        return FakeWithinKernelCUDAArray(ret)
    return ret