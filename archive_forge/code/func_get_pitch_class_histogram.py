import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def get_pitch_class_histogram(self, use_duration=False, use_velocity=False, normalize=False):
    """Computes the frequency of pitch classes of this instrument,
        optionally weighted by their durations or velocities.

        Parameters
        ----------
        use_duration : bool
            Weight frequency by note duration.
        use_velocity : bool
            Weight frequency by note velocity.
        normalize : bool
            Normalizes the histogram such that the sum of bin values is 1.

        Returns
        -------
        histogram : np.ndarray, shape=(12,)
            Histogram of pitch classes given current instrument, optionally
            weighted by their durations or velocities.
        """
    if self.is_drum:
        return np.zeros(12)
    weights = np.ones(len(self.notes))
    if use_duration:
        weights *= [note.end - note.start for note in self.notes]
    if use_velocity:
        weights *= [note.velocity for note in self.notes]
    histogram, _ = np.histogram([n.pitch % 12 for n in self.notes], bins=np.arange(13), weights=weights, density=normalize)
    return histogram