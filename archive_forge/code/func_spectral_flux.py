from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def spectral_flux(spectrogram, diff_frames=None):
    """
    Spectral Flux.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.

    Returns
    -------
    spectral_flux : numpy array
        Spectral flux onset detection function.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.

    """
    from madmom.audio.spectrogram import SpectrogramDifference
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames, positive_diffs=True)
    return np.asarray(np.sum(spectrogram, axis=1))