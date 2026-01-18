import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvvm": file_path
    - "libdevice": List[Tuple[arch, file_path]]
    - "cudalib_dir": directory_path

    Note: The result of the function is cached.
    """
    if hasattr(get_cuda_paths, '_cached_result'):
        return get_cuda_paths._cached_result
    else:
        d = {'nvvm': _get_nvvm_path(), 'libdevice': _get_libdevice_paths(), 'cudalib_dir': _get_cudalib_dir(), 'static_cudalib_dir': _get_static_cudalib_dir()}
        get_cuda_paths._cached_result = d
        return d