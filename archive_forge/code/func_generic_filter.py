import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def generic_filter(input, function, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    """Compute a multi-dimensional filter using the provided raw kernel or
    reduction kernel.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.ReductionKernel or cupy.RawKernel):
            The kernel or function to apply to each region.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. note::
        If the `function` is a :class:`cupy.RawKernel` then it must be for a
        function that has the following signature. Unlike most functions, this
        should not utilize `blockDim`/`blockIdx`/`threadIdx`::

            __global__ void func(double *buffer, int filter_size,
                                 double *return_value)

        If the `function` is a :class:`cupy.ReductionKernel` then it must be
        for a kernel that takes 1 array input and produces 1 'scalar' output.

    .. seealso:: :func:`scipy.ndimage.generic_filter`
    """
    _, footprint, _ = _filters_core._check_size_footprint_structure(input.ndim, size, footprint, None, 2, True)
    filter_size = int(footprint.sum())
    origins, int_type = _filters_core._check_nd_args(input, footprint, mode, origin, 'footprint')
    in_dtype = input.dtype
    sub = _filters_generic._get_sub_kernel(function)
    if footprint.size == 0:
        return cupy.zeros_like(input)
    output = _util._get_output(output, input)
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    args = (filter_size, mode, footprint.shape, offsets, float(cval), int_type)
    if isinstance(sub, cupy.RawKernel):
        kernel = _filters_generic._get_generic_filter_raw(sub, *args)
    elif isinstance(sub, cupy.ReductionKernel):
        kernel = _filters_generic._get_generic_filter_red(sub, in_dtype, output.dtype, *args)
    return _filters_core._call_kernel(kernel, input, footprint, output, weights_dtype=bool)