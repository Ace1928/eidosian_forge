from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def smooth_histogram(histogram, smooth):
    """
    Smooth the given histogram.

    Parameters
    ----------
    histogram : tuple
        Histogram (tuple of 2 numpy arrays, the first giving the strengths of
        the bins and the second corresponding delay values).
    smooth : int or numpy array
        Smoothing kernel (size).

    Returns
    -------
    histogram_bins : numpy array
        Bins of the smoothed histogram.
    histogram_delays : numpy array
        Corresponding delays.

    Notes
    -----
    If `smooth` is an integer, a Hamming window of that length will be used as
    a smoothing kernel.

    """
    return (smooth_signal(histogram[0], smooth), histogram[1])