import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def pitch_tuning(frequencies: ArrayLike, *, resolution: float=0.01, bins_per_octave: int=12) -> float:
    """Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.

    Parameters
    ----------
    frequencies : array-like, float
        A collection of frequencies detected in the signal.
        See `piptrack`
    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to cents.
    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    estimate_tuning : Estimating tuning from time-series or spectrogram input

    Examples
    --------
    >>> # Generate notes at +25 cents
    >>> freqs = librosa.cqt_frequencies(n_bins=24, fmin=55, tuning=0.25)
    >>> librosa.pitch_tuning(freqs)
    0.25

    >>> # Track frequencies from a real spectrogram
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr,
    ...                                                     fill_nan=True)
    >>> # Select out pitches with high energy
    >>> freqs = freqs[mags > np.median(mags)]
    >>> librosa.pitch_tuning(freqs)
    -0.07
    """
    frequencies = np.atleast_1d(frequencies)
    frequencies = frequencies[frequencies > 0]
    if not np.any(frequencies):
        warnings.warn('Trying to estimate tuning from empty frequency set.', stacklevel=2)
        return 0.0
    residual = np.mod(bins_per_octave * convert.hz_to_octs(frequencies), 1.0)
    residual[residual >= 0.5] -= 1.0
    bins = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)
    counts, tuning = np.histogram(residual, bins)
    tuning_est: float = tuning[np.argmax(counts)]
    return tuning_est