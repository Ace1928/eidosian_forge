import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def pitch_bend_to_semitones(pitch_bend, semitone_range=2.0):
    """Convert a MIDI pitch bend value (in the range ``[-8192, 8191]``) to the
    bend amount in semitones.

    Parameters
    ----------
    pitch_bend : int
        MIDI pitch bend amount, in ``[-8192, 8191]``.
    semitone_range : float
        Convert to +/- this semitone range.  Default is 2., which is the
        General MIDI standard +/-2 semitone range.

    Returns
    -------
    semitones : float
        Number of semitones corresponding to this pitch bend amount.

    """
    return semitone_range * pitch_bend / 8192.0