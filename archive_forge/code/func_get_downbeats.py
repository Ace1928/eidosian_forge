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
def get_downbeats(self, start_time=0.0):
    """Return a list of downbeat locations, according to MIDI tempo changes
        and time signature change events.

        Parameters
        ----------
        start_time : float
            Location of the first downbeat, in seconds.

        Returns
        -------
        downbeats : np.ndarray
            Downbeat locations, in seconds.

        """
    beats = self.get_beats(start_time)
    time_signatures = copy.deepcopy(self.time_signature_changes)
    if not time_signatures or time_signatures[0].time > start_time:
        time_signatures.insert(0, TimeSignature(4, 4, start_time))

    def index(array, value, default):
        """ Returns the first index of a value in an array, or `default` if
            the value doesn't appear in the array."""
        idx = np.flatnonzero(np.isclose(array, value))
        if idx.size > 0:
            return idx[0]
        else:
            return default
    downbeats = []
    end_beat_idx = 0
    for start_ts, end_ts in zip(time_signatures[:-1], time_signatures[1:]):
        start_beat_idx = index(beats, start_ts.time, 0)
        end_beat_idx = index(beats, end_ts.time, start_beat_idx)
        if start_ts.numerator % 3 == 0 and start_ts.numerator != 3:
            downbeats.append(beats[start_beat_idx:end_beat_idx:start_ts.numerator // 3])
        else:
            downbeats.append(beats[start_beat_idx:end_beat_idx:start_ts.numerator])
    final_ts = time_signatures[-1]
    start_beat_idx = index(beats, final_ts.time, end_beat_idx)
    if final_ts.numerator % 3 == 0 and final_ts.numerator != 3:
        downbeats.append(beats[start_beat_idx::final_ts.numerator // 3])
    else:
        downbeats.append(beats[start_beat_idx::final_ts.numerator])
    downbeats = np.concatenate(downbeats)
    return downbeats[downbeats >= start_time]