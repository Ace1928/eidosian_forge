from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
def syncthreads(self):
    if self.abort:
        raise RuntimeError('abort flag set on syncthreads call')
    self.syncthreads_blocked = True
    self.syncthreads_event.wait()
    self.syncthreads_event.clear()
    if self.abort:
        raise RuntimeError('abort flag set on syncthreads clear')