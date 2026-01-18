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
def get_current_bpm():
    """ Convenience function which computs the current BPM based on the
            current tempo change and time signature events """
    if self.time_signature_changes:
        return qpm_to_bpm(tempi[tempo_idx], self.time_signature_changes[ts_idx].numerator, self.time_signature_changes[ts_idx].denominator)
    else:
        return tempi[tempo_idx]