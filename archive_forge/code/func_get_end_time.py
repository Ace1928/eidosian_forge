import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def get_end_time(self):
    """Returns the time of the end of the events in this instrument.

        Returns
        -------
        end_time : float
            Time, in seconds, of the last event.

        """
    events = [n.end for n in self.notes] + [b.time for b in self.pitch_bends] + [c.time for c in self.control_changes]
    if len(events) == 0:
        return 0.0
    else:
        return max(events)