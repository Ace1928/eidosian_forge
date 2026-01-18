import warnings
from typing import Optional, Tuple, Union
import numpy as np
def hertz_to_octave(freq: Union[float, np.ndarray], tuning: Optional[float]=0.0, bins_per_octave: Optional[int]=12):
    """
    Convert frequency from hertz to fractional octave numbers.
    Adapted from *librosa*.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        tuning (`float`, defaults to `0.`):
            Tuning deviation from the Stuttgart pitch (A440) in (fractional) bins per octave.
        bins_per_octave (`int`, defaults to `12`):
            Number of bins per octave.

    Returns:
        `float` or `np.ndarray`: The frequencies on the octave scale.
    """
    stuttgart_pitch = 440.0 * 2.0 ** (tuning / bins_per_octave)
    octave = np.log2(freq / (float(stuttgart_pitch) / 16))
    return octave