from __future__ import absolute_import, division, print_function
import numpy as np
import mido
def load_midi(filename):
    """
    Load notes from a MIDI file.

    Parameters
    ----------
    filename: str
        MIDI file.

    Returns
    -------
    numpy array
        Notes ('onset time' 'note number' 'duration' 'velocity' 'channel')

    """
    return MIDIFile(filename).notes