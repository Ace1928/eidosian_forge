from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def stft_detrend(self, x: np.ndarray, detr: Callable[[np.ndarray], np.ndarray] | Literal['linear', 'constant'] | None, p0: int | None=None, p1: int | None=None, *, k_offset: int=0, padding: PAD_TYPE='zeros', axis: int=-1) -> np.ndarray:
    """Short-time Fourier transform with a trend being subtracted from each
        segment beforehand.

        If `detr` is set to 'constant', the mean is subtracted, if set to
        "linear", the linear trend is removed. This is achieved by calling
        :func:`scipy.signal.detrend`. If `detr` is a function, `detr` is
        applied to each segment.
        All other parameters have the same meaning as in `~ShortTimeFFT.stft`.

        Note that due to the detrending, the original signal cannot be
        reconstructed by the `~ShortTimeFFT.istft`.

        See Also
        --------
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        :meth:`~ShortTimeFFT.stft`: Short-time Fourier transform
                                   (without detrending).
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
    if isinstance(detr, str):
        detr = partial(detrend, type=detr)
    elif not (detr is None or callable(detr)):
        raise ValueError(f'Parameter detr={detr!r} is not a str, function or ' + 'None!')
    n = x.shape[axis]
    if not n >= (m2p := (self.m_num - self.m_num_mid)):
        e_str = f'len(x)={len(x)!r}' if x.ndim == 1 else f'of axis={axis!r} of {x.shape}'
        raise ValueError(f'{e_str} must be >= ceil(m_num/2) = {m2p}!')
    if x.ndim > 1:
        x = np.moveaxis(x, axis, -1)
    p0, p1 = self.p_range(n, p0, p1)
    S_shape_1d = (self.f_pts, p1 - p0)
    S_shape = x.shape[:-1] + S_shape_1d if x.ndim > 1 else S_shape_1d
    S = np.zeros(S_shape, dtype=complex)
    for p_, x_ in enumerate(self._x_slices(x, k_offset, p0, p1, padding)):
        if detr is not None:
            x_ = detr(x_)
        S[..., :, p_] = self._fft_func(x_ * self.win.conj())
    if x.ndim > 1:
        return np.moveaxis(S, -2, axis if axis >= 0 else axis - 1)
    return S