from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def complex_flux(spectrogram, diff_frames=None, diff_max_bins=3, temporal_filter=3, temporal_origin=0):
    """
    ComplexFlux.

    ComplexFlux is based on the SuperFlux, but adds an additional local group
    delay based tremolo suppression.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    diff_max_bins : int, optional
        Number of bins used for maximum filter.
    temporal_filter : int, optional
        Temporal maximum filtering of the local group delay [frames].
    temporal_origin : int, optional
        Origin of the temporal maximum filter.

    Returns
    -------
    complex_flux : numpy array
        ComplexFlux onset detection function.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer,
           "Local group delay based vibrato and tremolo suppression for onset
           detection",
           Proceedings of the 14th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.

    """
    lgd = np.abs(spectrogram.stft.phase().lgd()) / np.pi
    if temporal_filter > 0:
        lgd = maximum_filter(lgd, size=[temporal_filter, 1], origin=temporal_origin)
    try:
        mask = np.zeros_like(spectrogram)
        num_bins = lgd.shape[1]
        for b in range(mask.shape[1]):
            corner_bins = np.nonzero(spectrogram.filterbank[:, b])[0]
            start_bin = corner_bins[0] - 1
            stop_bin = corner_bins[-1] + 2
            if start_bin < 0:
                start_bin = 0
            if stop_bin > num_bins:
                stop_bin = num_bins
            mask[:, b] = np.amin(lgd[:, start_bin:stop_bin], axis=1)
    except AttributeError:
        mask = minimum_filter(lgd, size=[1, 3])
    diff = spectrogram.diff(diff_frames=diff_frames, diff_max_bins=diff_max_bins, positive_diffs=True)
    return np.asarray(np.sum(diff * mask, axis=1))