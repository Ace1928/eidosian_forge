from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
@_implements(_scipy_fft.rfftn)
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='R2C')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`scipy.fft.rfftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    func = _default_fft_func(x, s, axes, value_type='R2C')
    return func(x, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C', overwrite_x=overwrite_x, plan=plan)