from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def detect_tempo(histogram, fps):
    """
    Extract the tempo from the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding delay values).
    fps : float
        Frames per second.

    Returns
    -------
    tempi : numpy array
        Numpy array with the dominant tempi [bpm] (first column) and their
        relative strengths (second column).

    """
    from scipy.signal import argrelmax
    bins = histogram[0]
    tempi = 60.0 * fps / histogram[1]
    peaks = argrelmax(bins, mode='wrap')[0]
    if len(peaks) == 0:
        if len(bins):
            ret = np.asarray([tempi[len(bins) // 2], 1.0])
        else:
            ret = np.asarray([NO_TEMPO, 0.0])
    elif len(peaks) == 1:
        ret = np.asarray([tempi[peaks[0]], 1.0])
    else:
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        ret = np.asarray(list(zip(tempi[sorted_peaks], strengths)))
    return np.atleast_2d(ret)