import os
import sys
import ctypes
from numba.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
from numba.cuda.cudadrv.error import CudaSupportError
def get_libdevice():
    d = get_cuda_paths()
    paths = d['libdevice'].info
    return paths