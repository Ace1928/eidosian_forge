import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def get_onsets(self):
    """Get all onsets of all notes played by this instrument.
        May contain duplicates.

        Returns
        -------
        onsets : np.ndarray
                List of all note onsets.

        """
    onsets = []
    for note in self.notes:
        onsets.append(note.start)
    return np.sort(onsets)