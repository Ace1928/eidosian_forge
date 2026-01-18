from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
@_implements(_scipy_fft.hfft2)
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, *, plan=None):
    """Compute the FFT of a two-dimensional signal that has Hermitian symmetry.

    Args:
        x (cupy.ndarray): Array to be transformed.
        s (None or tuple of ints): Shape of the real output.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
            (This argument is currently not supported)
        plan (None): This argument is currently not supported.

    Returns:
        cupy.ndarray:
            The real result of the 2-D Hermitian complex real FFT.

    .. seealso:: :func:`scipy.fft.hfft2`
    """
    if plan is not None:
        raise NotImplementedError('hfft2 plan is currently not yet supported')
    return irfft2(x.conj(), s, axes, _swap_direction(norm))