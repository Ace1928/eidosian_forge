from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].

    Returns
    -------
    m : numpy array
        Frequencies in Mel [Mel].

    """
    return 1127.01048 * np.log(np.asarray(f) / 700.0 + 1.0)