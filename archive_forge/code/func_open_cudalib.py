import os
import sys
import ctypes
from numba.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
from numba.cuda.cudadrv.error import CudaSupportError
def open_cudalib(lib):
    path = get_cudalib(lib)
    return ctypes.CDLL(path)