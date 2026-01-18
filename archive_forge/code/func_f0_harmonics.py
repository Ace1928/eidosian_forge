import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence
def f0_harmonics(x: np.ndarray, *, f0: np.ndarray, freqs: np.ndarray, harmonics: ArrayLike, kind: str='linear', fill_value: float=0, axis: int=-2) -> np.ndarray:
    """Compute the energy at selected harmonics of a time-varying
    fundamental frequency.

    This function can be used to reduce a `frequency * time` representation
    to a `harmonic * time` representation, effectively normalizing out for
    the fundamental frequency.  The result can be used as a representation
    of timbre when f0 corresponds to pitch, or as a representation of
    rhythm when f0 corresponds to tempo.

    This function differs from `interp_harmonics`, which computes the
    harmonics of *all* frequencies.

    Parameters
    ----------
    x : np.ndarray [shape=(..., frequencies, n)]
        The input array (e.g., STFT magnitudes)
    f0 : np.ndarray [shape=(..., n)]
        The fundamental frequency (f0) of each frame in the input
        Shape should match ``x.shape[-1]``
    freqs : np.ndarray, shape=(x.shape[axis]) or shape=x.shape
        The frequency values corresponding to X's elements along the
        chosen axis.
        Frequencies can also be time-varying, e.g. as computed by
        `reassigned_spectrogram`, in which case the shape should
        match ``x``.
    harmonics : list-like, non-negative
        Harmonics to compute as ``harmonics[i] * f0``
        Values less than one (e.g., 1/2) correspond to sub-harmonics.
    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.
    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.
    axis : int
        The axis corresponding to frequency in ``x``

    Returns
    -------
    f0_harm : np.ndarray [shape=(..., len(harmonics), n)]
        Interpolated energy at each specified harmonic of the fundamental
        frequency for each time step.

    See Also
    --------
    interp_harmonics
    librosa.feature.tempogram_ratio

    Examples
    --------
    This example estimates the fundamental (f0), and then extracts the first
    12 harmonics

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> f0, voicing, voicing_p = librosa.pyin(y=y, sr=sr, fmin=200, fmax=700)
    >>> S = np.abs(librosa.stft(y))
    >>> freqs = librosa.fft_frequencies(sr=sr)
    >>> harmonics = np.arange(1, 13)
    >>> f0_harm = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics=harmonics)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax =plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='log', ax=ax[0])
    >>> times = librosa.times_like(f0)
    >>> for h in harmonics:
    ...     ax[0].plot(times, h * f0, label=f"{h}*f0")
    >>> ax[0].legend(ncols=4, loc='lower right')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(f0_harm, ref=np.max),
    ...                          x_axis='time', ax=ax[1])
    >>> ax[1].set_yticks(harmonics-1)
    >>> ax[1].set_yticklabels(harmonics)
    >>> ax[1].set(ylabel='Harmonics')
    """
    result: np.ndarray
    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        if not is_unique(freqs, axis=0):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)
        idx = np.isfinite(freqs)

        def _f_interps(data, f):
            interp = scipy.interpolate.interp1d(freqs[idx], data[idx], axis=0, bounds_error=False, copy=False, assume_sorted=False, kind=kind, fill_value=fill_value)
            return interp(f)
        xfunc = np.vectorize(_f_interps, signature='(f),(h)->(h)')
        result = xfunc(x.swapaxes(axis, -1), np.multiply.outer(f0, harmonics)).swapaxes(axis, -1)
    elif freqs.shape == x.shape:
        if not np.all(is_unique(freqs, axis=axis)):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)

        def _f_interpd(data, frequencies, f):
            idx = np.isfinite(frequencies)
            interp = scipy.interpolate.interp1d(frequencies[idx], data[idx], axis=0, bounds_error=False, copy=False, assume_sorted=False, kind=kind, fill_value=fill_value)
            return interp(f)
        xfunc = np.vectorize(_f_interpd, signature='(f),(f),(h)->(h)')
        result = xfunc(x.swapaxes(axis, -1), freqs.swapaxes(axis, -1), np.multiply.outer(f0, harmonics)).swapaxes(axis, -1)
    else:
        raise ParameterError(f'freqs.shape={freqs.shape} is incompatible with input shape={x.shape}')
    return np.nan_to_num(result, copy=False, nan=fill_value)