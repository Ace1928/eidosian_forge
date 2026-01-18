from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
def expand_notes(notes, duration=0.6, velocity=100):
    """
    Expand notes to include duration and velocity.

    The given duration and velocity is only used if they are not set already.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row. Expected columns:
        'note_time' 'note_number' ['duration' ['velocity']]
    duration : float, optional
        Note duration if not defined by `notes`.
    velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    notes : numpy array, shape (num_notes, 2)
        Notes (including note duration and velocity).

    """
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    rows, columns = notes.shape
    if columns == 4:
        return notes
    elif columns == 3:
        new_columns = np.ones((rows, 1)) * velocity
    elif columns == 2:
        new_columns = np.ones((rows, 2)) * velocity
        new_columns[:, 0] = duration
    else:
        raise ValueError('unable to handle `notes` with %d columns' % columns)
    notes = np.hstack((notes, new_columns))
    return notes