from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
def clicks(*, times: Optional[_SequenceLike[_FloatLike_co]]=None, frames: Optional[_SequenceLike[_IntLike_co]]=None, sr: float=22050, hop_length: int=512, click_freq: float=1000.0, click_duration: float=0.1, click: Optional[np.ndarray]=None, length: Optional[int]=None) -> np.ndarray:
    """Construct a "click track".

    This returns a signal with the signal ``click`` sound placed at
    each specified time.

    Parameters
    ----------
    times : np.ndarray or None
        times to place clicks, in seconds
    frames : np.ndarray or None
        frame indices to place clicks
    sr : number > 0
        desired sampling rate of the output signal
    hop_length : int > 0
        if positions are specified by ``frames``, the number of samples between frames.
    click_freq : float > 0
        frequency (in Hz) of the default click signal.  Default is 1KHz.
    click_duration : float > 0
        duration (in seconds) of the default click signal.  Default is 100ms.
    click : np.ndarray or None
        (optional) click signal sample to use instead of the default click.
        Multi-channel is supported.
    length : int > 0
        desired number of samples in the output signal

    Returns
    -------
    click_signal : np.ndarray
        Synthesized click signal.
        This will be monophonic by default, or match the number of channels to a provided ``click`` signal.

    Raises
    ------
    ParameterError
        - If neither ``times`` nor ``frames`` are provided.
        - If any of ``click_freq``, ``click_duration``, or ``length`` are out of range.

    Examples
    --------
    >>> # Sonify detected beat events
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> y_beats = librosa.clicks(frames=beats, sr=sr)

    >>> # Or generate a signal of the same length as y
    >>> y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))

    >>> # Or use timing instead of frame indices
    >>> times = librosa.frames_to_time(beats, sr=sr)
    >>> y_beat_times = librosa.clicks(times=times, sr=sr)

    >>> # Or with a click frequency of 880Hz and a 500ms sample
    >>> y_beat_times880 = librosa.clicks(times=times, sr=sr,
    ...                                  click_freq=880, click_duration=0.5)

    Display click waveform next to the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr)
    >>> librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax[0])
    >>> librosa.display.waveshow(y_beat_times, sr=sr, label='Beat clicks',
    ...                          ax=ax[1])
    >>> ax[1].legend()
    >>> ax[0].label_outer()
    >>> ax[0].set_title(None)
    """
    positions: np.ndarray
    if times is None:
        if frames is None:
            raise ParameterError('either "times" or "frames" must be provided')
        positions = frames_to_samples(frames, hop_length=hop_length)
    else:
        positions = time_to_samples(times, sr=sr)
    if click is not None:
        util.valid_audio(click, mono=False)
    else:
        if click_duration <= 0:
            raise ParameterError('click_duration must be strictly positive')
        if click_freq <= 0:
            raise ParameterError('click_freq must be strictly positive')
        angular_freq = 2 * np.pi * click_freq / float(sr)
        click = np.logspace(0, -10, num=int(sr * click_duration), base=2.0)
        click *= np.sin(angular_freq * np.arange(len(click)))
    if length is None:
        length = positions.max() + click.shape[-1]
    else:
        if length < 1:
            raise ParameterError('length must be a positive integer')
        positions = positions[positions < length]
    shape = list(click.shape)
    shape[-1] = length
    click_signal = np.zeros(shape, dtype=np.float32)
    for start in positions:
        end = start + click.shape[-1]
        if end >= length:
            click_signal[..., start:] += click[..., :length - start]
        else:
            click_signal[..., start:end] += click
    return click_signal