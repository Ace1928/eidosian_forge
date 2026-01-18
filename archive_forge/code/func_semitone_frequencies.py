from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def semitone_frequencies(fmin, fmax, fref=A4):
    """
    Returns frequencies separated by semitones.

    Parameters
    ----------
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    semitone_frequencies : numpy array
        Semitone frequencies [Hz].

    """
    return log_frequencies(12, fmin, fmax, fref)