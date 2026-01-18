from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
def quantize_notes(notes, fps, length=None, num_pitches=None, velocity=None):
    """
    Quantize the notes with the given resolution.

    Create a sparse 2D array with rows corresponding to points in time
    (according to `fps` and `length`), and columns to note pitches (according
    to `num_pitches`). The values of the array correspond to the velocity of a
    sounding note at a given point in time (based on the note pitch, onset,
    duration and velocity). If no values for `length` and `num_pitches` are
    given, they are inferred from `notes`.

    Parameters
    ----------
    notes : 2D numpy array
        Notes to be quantized. Expected columns:
        'note_time' 'note_number' ['duration' ['velocity']]
        If `notes` contains no 'duration' column, only the frame of the
        onset will be set. If `notes` has no velocity column, a velocity
        of 1 is assumed.
    fps : float
        Quantize with `fps` frames per second.
    length : int, optional
        Length of the returned array. If 'None', the length will be set
        according to the latest sounding note.
    num_pitches : int, optional
        Number of pitches of the returned array. If 'None', the number of
        pitches will be based on the highest pitch in the `notes` array.
    velocity : float, optional
        Use this velocity for all quantized notes. If set, the last column of
        `notes` (if present) will be ignored.

    Returns
    -------
    numpy array
        Quantized notes.

    """
    notes = np.array(np.array(notes).T, dtype=np.float, ndmin=2).T
    if notes.ndim != 2:
        raise ValueError('only 2-dimensional notes supported.')
    if notes.shape[1] < 2:
        raise ValueError('notes must have at least 2 columns.')
    note_onsets = notes[:, 0]
    note_numbers = notes[:, 1].astype(np.int)
    note_offsets = np.copy(note_onsets)
    if notes.shape[1] > 2:
        note_offsets += notes[:, 2]
    if notes.shape[1] > 3 and velocity is None:
        note_velocities = notes[:, 3]
    else:
        velocity = velocity or 1
        note_velocities = np.ones(len(notes)) * velocity
    if length is None:
        length = int(round(np.max(note_offsets) * float(fps))) + 1
    if num_pitches is None:
        num_pitches = int(np.max(note_numbers)) + 1
    quantized = np.zeros((length, num_pitches))
    note_onsets = np.round(note_onsets * fps).astype(np.int)
    note_offsets = np.round(note_offsets * fps).astype(np.int) + 1
    for n, note in enumerate(notes):
        if num_pitches > note_numbers[n] >= 0:
            quantized[note_onsets[n]:note_offsets[n], note_numbers[n]] = note_velocities[n]
    return quantized