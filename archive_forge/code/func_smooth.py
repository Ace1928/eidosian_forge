from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def smooth(signal, kernel):
    """
    Smooth the signal along its first axis.

    Parameters
    ----------
    signal : numpy array
        Signal to be smoothed.
    kernel : numpy array or int
        Smoothing kernel (size).

    Returns
    -------
    numpy array
        Smoothed signal.

    Notes
    -----
    If `kernel` is an integer, a Hamming window of that length will be used
    as a smoothing kernel.

    """
    if kernel is None:
        return signal
    elif isinstance(kernel, integer_types):
        if kernel == 0:
            return signal
        elif kernel > 1:
            kernel = np.hamming(kernel)
        else:
            raise ValueError("can't create a smoothing kernel of size %d" % kernel)
    elif isinstance(kernel, np.ndarray):
        kernel = kernel
    else:
        raise ValueError("can't smooth signal with %s" % kernel)
    if signal.ndim == 1:
        return np.convolve(signal, kernel, 'same')
    elif signal.ndim == 2:
        from scipy.signal import convolve2d
        return convolve2d(signal, kernel[:, np.newaxis], 'same')
    else:
        raise ValueError('signal must be either 1D or 2D')