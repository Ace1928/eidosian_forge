from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def bins2frequencies(bins, bin_frequencies):
    """
    Convert bins to the corresponding frequencies.

    Parameters
    ----------
    bins : numpy array
        Bins (e.g. FFT bins).
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].

    Returns
    -------
    f : numpy array
        Corresponding frequencies [Hz].

    """
    return np.asarray(bin_frequencies, dtype=np.float)[np.asarray(bins)]