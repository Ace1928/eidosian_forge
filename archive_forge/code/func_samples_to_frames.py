from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def samples_to_frames(samples: _ScalarOrSequence[_IntLike_co], *, hop_length: int=512, n_fft: Optional[int]=None) -> Union[np.integer[Any], np.ndarray]:
    """Convert sample indices into STFT frames.

    Examples
    --------
    >>> # Get the frame numbers for every 256 samples
    >>> librosa.samples_to_frames(np.arange(0, 22050, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : int or np.ndarray [shape=(n,)]
        sample index or vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``- n_fft // 2``
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : int or np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times::

            frames[i] = floor( samples[i] / hop_length )

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)
    samples = np.asanyarray(samples)
    return np.asarray(np.floor((samples - offset) // hop_length), dtype=int)