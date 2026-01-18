import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def get_pitch_class_transition_matrix(self, normalize=False, time_thresh=0.05):
    """Computes the pitch class transition matrix of this instrument.
        Transitions are added whenever the end of a note is within
        ``time_tresh`` from the start of any other note.

        Parameters
        ----------
        normalize : bool
            Normalize transition matrix such that matrix sum equals to 1.
        time_thresh : float
            Maximum temporal threshold, in seconds, between the start of a note
            and end time of any other note for a transition to be added.

        Returns
        -------
        transition_matrix : np.ndarray, shape=(12,12)
            Pitch class transition matrix.
        """
    if self.is_drum or len(self.notes) <= 1:
        return np.zeros((12, 12))
    starts, ends, nodes = np.array([[x.start, x.end, x.pitch % 12] for x in self.notes]).T
    dist_mat = np.subtract.outer(ends, starts)
    sources, targets = np.where(abs(dist_mat) < time_thresh)
    transition_matrix, _, _ = np.histogram2d(nodes[sources], nodes[targets], bins=np.arange(13), normed=normalize)
    return transition_matrix