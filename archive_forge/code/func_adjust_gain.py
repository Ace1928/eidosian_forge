from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def adjust_gain(signal, gain):
    """"
    Adjust the gain of the signal.

    Parameters
    ----------
    signal : numpy array
        Signal to be adjusted.
    gain : float
        Gain adjustment level [dB].

    Returns
    -------
    numpy array
        Signal with adjusted gain.

    Notes
    -----
    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    `gain` values > 0 amplify the signal and are only supported for signals
    with float dtype to prevent clipping and integer overflows.

    """
    gain = np.power(np.sqrt(10.0), 0.1 * gain)
    if gain > 1 and np.issubdtype(signal.dtype, np.integer):
        raise ValueError('positive gain adjustments are only supported for float dtypes.')
    return np.asanyarray(signal * gain, dtype=signal.dtype)