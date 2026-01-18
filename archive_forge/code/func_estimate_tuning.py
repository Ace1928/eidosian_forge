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
def estimate_tuning(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: Optional[int]=2048, resolution: float=0.01, bins_per_octave: int=12, **kwargs: Any) -> float:
    """Estimate the tuning of an audio time series or spectrogram input.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio signal. Multi-channel is supported..
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        magnitude or power spectrogram
    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if ``y`` is provided.
    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to measurements in cents.
    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave
    **kwargs : additional keyword arguments
        Additional arguments passed to `piptrack`

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin).

        Note that if multichannel input is provided, a single tuning estimate is provided spanning all
        channels.

    See Also
    --------
    piptrack : Pitch tracking by parabolic interpolation

    Examples
    --------
    With time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.estimate_tuning(y=y, sr=sr)
    -0.08000000000000002

    In tenths of a cent

    >>> librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)
    -0.016000000000000014

    Using spectrogram input

    >>> S = np.abs(librosa.stft(y))
    >>> librosa.estimate_tuning(S=S, sr=sr)
    -0.08000000000000002

    Using pass-through arguments to `librosa.piptrack`

    >>> librosa.estimate_tuning(y=y, sr=sr, n_fft=8192,
    ...                         fmax=librosa.note_to_hz('G#9'))
    -0.08000000000000002
    """
    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    pitch_mask = pitch > 0
    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0
    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask], resolution=resolution, bins_per_octave=bins_per_octave)