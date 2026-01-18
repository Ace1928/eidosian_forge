import functools
import math
import operator
import textwrap
import cupy
@cupy.memoize(for_each_device=True)
def get_raw_spline1d_kernel(axis, ndim, mode, order, index_type='int', data_type='double', pole_type='double', block_size=128):
    """Generate a kernel for applying a spline prefilter along a given axis."""
    poles = get_poles(order)
    largest_pole = max([abs(p) for p in poles])
    tol = 1e-10 if pole_type == 'float' else 1e-18
    n_boundary = math.ceil(math.log(tol, largest_pole))
    code = _FILTER_GENERAL.format(index_type=index_type, data_type=data_type, pole_type=pole_type)
    code += _get_spline1d_code(mode, poles, n_boundary)
    mode_str = mode.replace('-', '_')
    kernel_name = f'cupyx_scipy_ndimage_spline_filter_{ndim}d_ord{order}_axis{axis}_{mode_str}'
    code += _batch_spline1d_strided_template.format(ndim=ndim, axis=axis, block_size=block_size, kernel_name=kernel_name)
    return cupy.RawKernel(code, kernel_name)