import math
import numbers
import os
import cupy
from ._util import _get_inttype
@cupy.memoize(True)
def get_pba2d_src(block_size_2d=64, marker=-32768, pixel_int2_t='short2'):
    make_pixel_func = 'make_' + pixel_int2_t
    pba2d_code = pba2d_defines_template.format(block_size_2d=block_size_2d, marker=marker, pixel_int2_t=pixel_int2_t, make_pixel_func=make_pixel_func)
    kernel_directory = os.path.join(os.path.dirname(__file__), 'cuda')
    with open(os.path.join(kernel_directory, 'pba_kernels_2d.h'), 'rt') as f:
        pba2d_kernels = '\n'.join(f.readlines())
    pba2d_code += pba2d_kernels
    return pba2d_code