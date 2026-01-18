from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def fourier_tempo_frequencies(*, sr: float=22050, win_length: int=384, hop_length: int=512) -> np.ndarray:
    """Compute the frequencies (in beats per minute) corresponding
    to a Fourier tempogram matrix.

    Parameters
    ----------
    sr : number > 0
        The audio sampling rate
    win_length : int > 0
        The number of frames per analysis window
    hop_length : int > 0
        The number of samples between each bin

    Returns
    -------
    bin_frequencies : ndarray [shape=(win_length // 2 + 1 ,)]
        vector of bin frequencies measured in BPM.

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.fourier_tempo_frequencies(win_length=384)
    array([ 0.   ,  0.117,  0.234, ..., 22.266, 22.383, 22.5  ])
    """
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)