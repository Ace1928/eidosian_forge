from numpy import prod
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._fft import (_convert_fft_type, _default_fft_func, _fft,
from cupy.fft._cache import get_plan_cache
def get_fft_plan(a, shape=None, axes=None, value_type='C2C'):
    """ Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        a (cupy.ndarray): Array to be transform, assumed to be either C- or
            F- contiguous.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (None or int or tuple of int):  The axes of the array to
            transform. If `None`, it is assumed that all axes are transformed.

            Currently, for performing N-D transform these must be a set of up
            to three adjacent axes, and must include either the first or the
            last axis of the array.
        value_type (str): The FFT type to perform. Acceptable values are:

            * 'C2C': complex-to-complex transform (default)
            * 'R2C': real-to-complex transform
            * 'C2R': complex-to-real transform

    Returns:
        a cuFFT plan for either 1D transform (``cupy.cuda.cufft.Plan1d``) or
        N-D transform (``cupy.cuda.cufft.PlanNd``).

    .. note::
        The returned plan can not only be passed as one of the arguments of
        the functions in ``cupyx.scipy.fftpack``, but also be used as a
        context manager for both ``cupy.fft`` and ``cupyx.scipy.fftpack``
        functions:

        .. code-block:: python

            x = cupy.random.random(16).reshape(4, 4).astype(complex)
            plan = cupyx.scipy.fftpack.get_fft_plan(x)
            with plan:
                y = cupy.fft.fftn(x)
                # alternatively:
                y = cupyx.scipy.fftpack.fftn(x)  # no explicit plan is given!
            # alternatively:
            y = cupyx.scipy.fftpack.fftn(x, plan=plan)  # pass plan explicitly

        In the first case, no cuFFT plan will be generated automatically,
        even if ``cupy.fft.config.enable_nd_planning = True`` is set.

    .. note::
        If this function is called under the context of
        :func:`~cupy.fft.config.set_cufft_callbacks`, the generated plan will
        have callbacks enabled.

    .. warning::
        This API is a deviation from SciPy's, is currently experimental, and
        may be changed in the future version.
    """
    if a.flags.c_contiguous:
        order = 'C'
    elif a.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('Input array a must be contiguous')
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(axes, int):
        axes = (axes,)
    if shape is not None and axes is not None and (len(shape) != len(axes)):
        raise ValueError('Shape and axes have different lengths.')
    if axes is None:
        n = a.ndim if shape is None else len(shape)
        axes = tuple((i for i in range(-n, 0)))
        if n == 1:
            axis1D = 0
    else:
        n = len(axes)
        if n == 1:
            axis1D = axes[0]
            if axis1D >= a.ndim or axis1D < -a.ndim:
                err = 'The chosen axis ({0}) exceeds the number of dimensions of a ({1})'.format(axis1D, a.ndim)
                raise ValueError(err)
        elif n > 3:
            raise ValueError('Only up to three axes is supported')
    transformed_shape = shape
    shape = list(a.shape)
    if transformed_shape is not None:
        for s, axis in zip(transformed_shape, axes):
            if s is not None:
                if axis == axes[-1] and value_type == 'C2R':
                    s = s // 2 + 1
                shape[axis] = s
    shape = tuple(shape)
    out_dtype = _output_dtype(a.dtype, value_type)
    fft_type = _convert_fft_type(out_dtype, value_type)
    if n > 1 and value_type != 'C2C' and a.flags.f_contiguous:
        raise ValueError('C2R/R2C PlanNd for F-order arrays is not supported')
    if n > 1:
        if cupy.cuda.runtime.is_hip and value_type == 'C2R':
            raise RuntimeError("hipFFT's C2R PlanNd is buggy and unsupported")
        out_size = _get_fftn_out_size(shape, transformed_shape, axes[-1], value_type)
        plan = _get_cufft_plan_nd(shape, fft_type, axes=axes, order=order, out_size=out_size, to_cache=False)
    else:
        if value_type != 'C2R':
            out_size = shape[axis1D]
        else:
            out_size = _get_fftn_out_size(shape, transformed_shape, axis1D, value_type)
        batch = prod(shape) // shape[axis1D]
        devices = None if not config.use_multi_gpus else config._devices
        keys = (out_size, fft_type, batch, devices)
        mgr = config.get_current_callback_manager()
        if mgr is not None:
            load_aux = mgr.cb_load_aux_arr
            store_aux = mgr.cb_store_aux_arr
            keys += (mgr.cb_load, mgr.cb_store, 0 if load_aux is None else load_aux.data.ptr, 0 if store_aux is None else store_aux.data.ptr)
        cache = get_plan_cache()
        cached_plan = cache.get(keys)
        if cached_plan is not None:
            plan = cached_plan
        elif mgr is None:
            plan = cufft.Plan1d(out_size, fft_type, batch, devices=devices)
        else:
            if devices:
                raise NotImplementedError('multi-GPU cuFFT callbacks are not yet supported')
            plan = mgr.create_plan(('Plan1d', keys[:-3]))
            mgr.set_callbacks(plan)
    return plan