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
def adjust_events(event_getter):
    """ This function calls event_getter with each instrument as the
            sole argument and adjusts the events which are returned."""
    for instrument in self.instruments:
        event_getter(instrument).sort(key=lambda e: e.time)
    event_times = np.array([event.time for instrument in self.instruments for event in event_getter(instrument)])
    adjusted_event_times = np.interp(event_times, original_times, new_times)
    for n, event in enumerate([event for instrument in self.instruments for event in event_getter(instrument)]):
        event.time = adjusted_event_times[n]
    for instrument in self.instruments:
        valid_events = [event for event in event_getter(instrument) if event.time == new_times[0]]
        if valid_events:
            valid_events = valid_events[-1:]
        valid_events.extend((event for event in event_getter(instrument) if event.time > new_times[0] and event.time < new_times[-1]))
        event_getter(instrument)[:] = valid_events