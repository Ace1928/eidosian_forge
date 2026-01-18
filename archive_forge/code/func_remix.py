from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def remix(signal, num_channels):
    """
    Remix the signal to have the desired number of channels.

    Parameters
    ----------
    signal : numpy array
        Signal to be remixed.
    num_channels : int
        Number of channels.

    Returns
    -------
    numpy array
        Remixed signal (same dtype as `signal`).

    Notes
    -----
    This function does not support arbitrary channel number conversions.
    Only down-mixing to and up-mixing from mono signals is supported.

    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    If the signal should be down-mixed to mono and has an integer dtype, it
    will be converted to float internally and then back to the original dtype
    to prevent clipping of the signal. To avoid this double conversion,
    convert the dtype first.

    """
    if num_channels == signal.ndim or num_channels is None:
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        return np.mean(signal, axis=-1).astype(signal.dtype)
    elif num_channels > 1 and signal.ndim == 1:
        return np.tile(signal[:, np.newaxis], num_channels)
    else:
        raise NotImplementedError('Requested %d channels, but got %d channels and channel conversion is not implemented.' % (num_channels, signal.shape[1]))