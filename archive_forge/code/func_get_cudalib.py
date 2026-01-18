import os
import sys
import ctypes
from numba.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
from numba.cuda.cudadrv.error import CudaSupportError
def get_cudalib(lib, static=False):
    """
    Find the path of a CUDA library based on a search of known locations. If
    the search fails, return a generic filename for the library (e.g.
    'libnvvm.so' for 'nvvm') so that we may attempt to load it using the system
    loader's search mechanism.
    """
    if lib == 'nvvm':
        return get_cuda_paths()['nvvm'].info or _dllnamepattern % 'nvvm'
    else:
        dir_type = 'static_cudalib_dir' if static else 'cudalib_dir'
        libdir = get_cuda_paths()[dir_type].info
    candidates = find_lib(lib, libdir, static=static)
    namepattern = _staticnamepattern if static else _dllnamepattern
    return max(candidates) if candidates else namepattern % lib