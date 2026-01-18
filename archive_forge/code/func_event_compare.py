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
def event_compare(event1, event2):
    """Compares two events for sorting.

            Events are sorted by tick time ascending. Events with the same tick
            time ares sorted by event type. Some events are sorted by
            additional values. For example, Note On events are sorted by pitch
            then velocity, ensuring that a Note Off (Note On with velocity 0)
            will never follow a Note On with the same pitch.

            Parameters
            ----------
            event1, event2 : mido.Message
               Two events to be compared.
            """
    secondary_sort = {'set_tempo': lambda e: 1 * 256 * 256, 'time_signature': lambda e: 2 * 256 * 256, 'key_signature': lambda e: 3 * 256 * 256, 'lyrics': lambda e: 4 * 256 * 256, 'text_events': lambda e: 5 * 256 * 256, 'program_change': lambda e: 6 * 256 * 256, 'pitchwheel': lambda e: 7 * 256 * 256 + e.pitch, 'control_change': lambda e: 8 * 256 * 256 + e.control * 256 + e.value, 'note_off': lambda e: 9 * 256 * 256 + e.note * 256, 'note_on': lambda e: 10 * 256 * 256 + e.note * 256 + e.velocity, 'end_of_track': lambda e: 11 * 256 * 256}
    if event1.time == event2.time and event1.type in secondary_sort and (event2.type in secondary_sort):
        return secondary_sort[event1.type](event1) - secondary_sort[event2.type](event2)
    return event1.time - event2.time