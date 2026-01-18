import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
@cupy.memoize(True)
def get_pba3d_src(block_size_3d=32, marker=-2147483648, max_int=2147483647, size_max=1024):
    pba3d_code = pba3d_defines_template.format(block_size_3d=block_size_3d, marker=marker, max_int=max_int)
    if size_max > 1024:
        pba3d_code += pba3d_defines_encode_64bit
    else:
        pba3d_code += pba3d_defines_encode_32bit
    kernel_directory = os.path.join(os.path.dirname(__file__), 'cuda')
    with open(os.path.join(kernel_directory, 'pba_kernels_3d.h'), 'rt') as f:
        pba3d_kernels = '\n'.join(f.readlines())
    pba3d_code += pba3d_kernels
    return pba3d_code