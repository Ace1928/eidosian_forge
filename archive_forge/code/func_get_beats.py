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
def get_beats(self, start_time=0.0):
    """Returns a list of beat locations, according to MIDI tempo changes.
        For compound meters (any whose numerator is a multiple of 3 greater
        than 3), this method returns every third denominator note (for 6/8
        or 6/16 time, for example, it will return every third 8th note or
        16th note, respectively). For all other meters, this method returns
        every denominator note (every quarter note for 3/4 or 4/4 time, for
        example).

        Parameters
        ----------
        start_time : float
            Location of the first beat, in seconds.

        Returns
        -------
        beats : np.ndarray
            Beat locations, in seconds.

        """
    tempo_change_times, tempi = self.get_tempo_changes()
    beats = [start_time]
    tempo_idx = 0
    while tempo_idx < tempo_change_times.shape[0] - 1 and beats[-1] > tempo_change_times[tempo_idx + 1]:
        tempo_idx += 1
    self.time_signature_changes.sort(key=lambda ts: ts.time)
    ts_idx = 0
    while ts_idx < len(self.time_signature_changes) - 1 and beats[-1] >= self.time_signature_changes[ts_idx + 1].time:
        ts_idx += 1

    def get_current_bpm():
        """ Convenience function which computs the current BPM based on the
            current tempo change and time signature events """
        if self.time_signature_changes:
            return qpm_to_bpm(tempi[tempo_idx], self.time_signature_changes[ts_idx].numerator, self.time_signature_changes[ts_idx].denominator)
        else:
            return tempi[tempo_idx]

    def gt_or_close(a, b):
        """ Returns True if a > b or a is close to b """
        return a > b or np.isclose(a, b)
    end_time = self.get_end_time()
    while beats[-1] < end_time:
        bpm = get_current_bpm()
        next_beat = beats[-1] + 60.0 / bpm
        if tempo_idx < tempo_change_times.shape[0] - 1 and next_beat > tempo_change_times[tempo_idx + 1]:
            next_beat = beats[-1]
            beat_remaining = 1.0
            while tempo_idx < tempo_change_times.shape[0] - 1 and next_beat + beat_remaining * 60.0 / bpm >= tempo_change_times[tempo_idx + 1]:
                overshot_ratio = (tempo_change_times[tempo_idx + 1] - next_beat) / (60.0 / bpm)
                next_beat += overshot_ratio * 60.0 / bpm
                beat_remaining -= overshot_ratio
                tempo_idx = tempo_idx + 1
                bpm = get_current_bpm()
            next_beat += beat_remaining * 60.0 / bpm
        if self.time_signature_changes and ts_idx == 0:
            current_ts_time = self.time_signature_changes[ts_idx].time
            if current_ts_time > beats[-1] and gt_or_close(next_beat, current_ts_time):
                next_beat = current_ts_time
        if ts_idx < len(self.time_signature_changes) - 1:
            next_ts_time = self.time_signature_changes[ts_idx + 1].time
            if gt_or_close(next_beat, next_ts_time):
                next_beat = next_ts_time
                ts_idx += 1
                bpm = get_current_bpm()
        beats.append(next_beat)
    beats = np.array(beats[:-1])
    return beats