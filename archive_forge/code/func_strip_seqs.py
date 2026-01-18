import os
import re
import sys
import time
import codecs
import locale
import select
import struct
import platform
import warnings
import functools
import contextlib
import collections
from .color import COLOR_DISTANCE_ALGORITHMS
from .keyboard import (_time_left,
from .sequences import Termcap, Sequence, SequenceTextWrapper
from .colorspace import RGB_256TABLE
from .formatters import (COLORS,
from ._capabilities import CAPABILITY_DATABASE, CAPABILITIES_ADDITIVES, CAPABILITIES_RAW_MIXIN
def strip_seqs(self, text):
    """
        Return ``text`` stripped of only its terminal sequences.

        :rtype: str
        :returns: Text with terminal sequences removed

        >>> term.strip_seqs(u'\\x1b[0;3mxyz')
        u'xyz'
        >>> term.strip_seqs(term.cuf(5) + term.red(u'test'))
        u'     test'

        .. note:: Non-destructive sequences that adjust horizontal distance
            (such as ``\\b`` or ``term.cuf(5)``) are replaced by destructive
            space or erasing.
        """
    return Sequence(text, self).strip_seqs()