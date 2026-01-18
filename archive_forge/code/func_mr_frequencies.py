import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from numba import jit
from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated
from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies
from numpy.typing import ArrayLike, DTypeLike
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@cache(level=10)
def mr_frequencies(tuning: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate center frequencies and sample rate pairs.

    This function will return center frequency and corresponding sample rates
    to obtain similar pitch filterbank settings as described in [#]_.
    Instead of starting with MIDI pitch `A0`, we start with `C0`.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    tuning : float [scalar]
        Tuning deviation from A440, measure as a fraction of the equally
        tempered semitone (1/12 of an octave).

    Returns
    -------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter, used for multirate filterbank.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    librosa.filters.semitone_filterbank
    """
    center_freqs = midi_to_hz(np.arange(24 + tuning, 109 + tuning))
    sample_rates = np.asarray(len(np.arange(0, 36)) * [882.0] + len(np.arange(36, 70)) * [4410.0] + len(np.arange(70, 85)) * [22050.0])
    return (center_freqs, sample_rates)