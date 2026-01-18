from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def estimate_beat_start(self, candidates=10, tolerance=0.025):
    """Estimate the location of the first beat based on which of the first
        few onsets results in the best correlation with the onset spike train.

        Parameters
        ----------
        candidates : int
            Number of candidate onsets to try.
        tolerance : float
            The tolerance in seconds around which onsets will be used to
            treat a beat as correct.

        Returns
        -------
        beat_start : float
            The offset which is chosen as the beat start location.
        """
    note_list = [n for i in self.instruments for n in i.notes]
    if not note_list:
        raise ValueError("Can't estimate beat start when there are no notes.")
    note_list.sort(key=lambda note: note.start)
    beat_candidates = []
    start_times = []
    onset_index = 0
    while len(beat_candidates) <= candidates and len(beat_candidates) <= len(note_list) and (onset_index < len(note_list)):
        if onset_index == 0 or np.abs(note_list[onset_index - 1].start - note_list[onset_index].start) > 0.001:
            beat_candidates.append(self.get_beats(note_list[onset_index].start))
            start_times.append(note_list[onset_index].start)
        onset_index += 1
    onset_scores = np.zeros(len(beat_candidates))
    fs = 1000
    onset_signal = np.zeros(int(fs * (self.get_end_time() + 1)))
    for note in note_list:
        onset_signal[int(note.start * fs)] += note.velocity
    for n, beats in enumerate(beat_candidates):
        beat_signal = np.zeros(int(fs * (self.get_end_time() + 1)))
        for beat in np.append(0, beats):
            if beat - tolerance < 0:
                beat_window = np.ones(int(fs * 2 * tolerance + (beat - tolerance) * fs))
                beat_signal[:int((beat + tolerance) * fs)] = beat_window
            else:
                beat_start = int((beat - tolerance) * fs)
                beat_end = beat_start + int(fs * tolerance * 2)
                beat_window = np.ones(int(fs * tolerance * 2))
                beat_signal[beat_start:beat_end] = beat_window
        onset_scores[n] = np.dot(beat_signal, onset_signal) / beats.shape[0]
    return start_times[np.argmax(onset_scores)]