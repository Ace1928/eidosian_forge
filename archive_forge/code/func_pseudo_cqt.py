import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
@cache(level=20)
def pseudo_cqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    """Compute the pseudo constant-Q transform of an audio signal.

    This uses a single fft size that is the smallest power of 2 that is greater
    than or equal to the max of:

        1. The longest CQT filter
        2. 2x the hop_length

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    dtype : np.dtype, optional
        The complex data type for CQT calculations.
        By default, this is inferred to match the precision of the input signal.

    Returns
    -------
    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]
        Pseudo Constant-Q energy for each frequency at each time.

    Notes
    -----
    This function caches at level 20.
    """
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    lengths, _ = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha)
    fft_basis, n_fft, _ = __vqt_filter_fft(sr, freqs, filter_scale, norm, sparsity, hop_length=hop_length, window=window, dtype=dtype, alpha=alpha)
    fft_basis = np.abs(fft_basis)
    C: np.ndarray = __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, window='hann', dtype=dtype, phase=False)
    if scale:
        C /= np.sqrt(n_fft)
    else:
        lengths = util.expand_to(lengths, ndim=C.ndim, axes=-2)
        C *= np.sqrt(lengths / n_fft)
    return C