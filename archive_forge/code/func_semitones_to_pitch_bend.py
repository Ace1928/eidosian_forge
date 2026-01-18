import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def semitones_to_pitch_bend(semitones, semitone_range=2.0):
    """Convert a semitone value to the corresponding MIDI pitch bend integer.

    Parameters
    ----------
    semitones : float
        Number of semitones for the pitch bend.
    semitone_range : float
        Convert to +/- this semitone range.  Default is 2., which is the
        General MIDI standard +/-2 semitone range.

    Returns
    -------
    pitch_bend : int
        MIDI pitch bend amount, in ``[-8192, 8191]``.

    """
    return int(8192 * (semitones / semitone_range))