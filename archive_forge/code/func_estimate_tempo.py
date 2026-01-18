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
def estimate_tempo(self):
    """Returns the best tempo estimate from
        :func:`pretty_midi.PrettyMIDI.estimate_tempi()`, for convenience.

        Returns
        -------
        tempo : float
            Estimated tempo, in bpm

        """
    tempi = self.estimate_tempi()[0]
    if tempi.size == 0:
        raise ValueError("Can't provide a global tempo estimate when there are fewer than two notes.")
    return tempi[0]