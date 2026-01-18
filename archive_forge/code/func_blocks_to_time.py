from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def blocks_to_time(blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int, hop_length: int, sr: int) -> Union[np.floating[Any], np.ndarray]:
    """Convert block indices to time (in seconds)

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames
    sr : int > 0
        The sampling rate (samples per second)

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        The time index or indices (in seconds) corresponding to the
        beginning of each provided block.

        Note that these correspond to the time of the *first* sample
        in each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_samples

    Examples
    --------
    Get time indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_time = librosa.blocks_to_time(n, block_length=16,
    ...                                     hop_length=512, sr=sr)

    """
    samples = blocks_to_samples(blocks, block_length=block_length, hop_length=hop_length)
    return samples_to_time(samples, sr=sr)