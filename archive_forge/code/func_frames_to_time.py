from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def frames_to_time(frames: _ScalarOrSequence[_IntLike_co], *, sr: float=22050, hop_length: int=512, n_fft: Optional[int]=None) -> Union[np.floating[Any], np.ndarray]:
    """Convert frame counts to time (seconds).

    Parameters
    ----------
    frames : np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number::

            times[i] = frames[i] * hop_length / sr

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr)
    """
    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples_to_time(samples, sr=sr)