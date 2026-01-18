from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def modified_kullback_leibler(spectrogram, diff_frames=1, epsilon=EPSILON):
    """
    Modified Kullback-Leibler.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.
    diff_frames : int, optional
        Number of frames to calculate the diff to.
    epsilon : float, optional
        Add `epsilon` to the `spectrogram` avoid division by 0.

    Returns
    -------
    modified_kullback_leibler : numpy array
         MKL onset detection function.

    Notes
    -----
    The implementation presented in [1]_ is used instead of the original work
    presented in [2]_.

    References
    ----------
    .. [1] Paul Brossier,
           "Automatic Annotation of Musical Audio for Interactive
           Applications",
           PhD thesis, Queen Mary University of London, 2006.
    .. [2] Stephen Hainsworth and Malcolm Macleod,
           "Onset Detection in Musical Audio Signals",
           Proceedings of the International Computer Music Conference (ICMC),
           2003.

    """
    if epsilon <= 0:
        raise ValueError('a positive value must be added before division')
    mkl = np.zeros_like(spectrogram)
    mkl[diff_frames:] = spectrogram[diff_frames:] / (spectrogram[:-diff_frames] + epsilon)
    return np.asarray(np.mean(np.log(1 + mkl), axis=1))