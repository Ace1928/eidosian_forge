from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
@_implements(_scipy_fft.irfftn)
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='C2R')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            ``2*(m-1)`` where `m` is the length of the final transformed axis
            of the input.

    .. seealso:: :func:`scipy.fft.irfftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    if 10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010 and int(cupy.cuda.device.get_compute_capability()) < 70 and (_size_last_transform_axis(x.shape, s, axes) == 2):
        warnings.warn('Output of irfftn might not be correct due to issue of cuFFT in CUDA 10.1/10.2 on Pascal or older GPUs.')
    func = _default_fft_func(x, s, axes, value_type='C2R')
    return func(x, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R', overwrite_x=overwrite_x, plan=plan)