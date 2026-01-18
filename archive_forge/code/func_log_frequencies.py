from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """
    Returns frequencies aligned on a logarithmic frequency scale.

    Parameters
    ----------
    bands_per_octave : int
        Number of filter bands per octave.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency [Hz].

    Returns
    -------
    log_frequencies : numpy array
        Logarithmically spaced frequencies [Hz].

    Notes
    -----
    If `bands_per_octave` = 12 and `fref` = 440 are used, the frequencies are
    equivalent to MIDI notes.

    """
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    return frequencies