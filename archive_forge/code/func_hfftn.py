from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
@_implements(_scipy_fft.hfftn)
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the FFT of a N-dimensional signal that has Hermitian symmetry.

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
            The real result of the N-D Hermitian complex real FFT.

    .. seealso:: :func:`scipy.fft.hfftn`
    """
    if plan is not None:
        raise NotImplementedError('hfftn plan is currently not yet supported')
    return irfftn(x.conj(), s, axes, _swap_direction(norm))