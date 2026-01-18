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
def split_seqs(self, text, maxsplit=0):
    """
        Return ``text`` split by individual character elements and sequences.

        :arg str text: String containing sequences
        :arg int maxsplit: When maxsplit is nonzero, at most maxsplit splits
            occur, and the remainder of the string is returned as the final element
            of the list (same meaning is argument for :func:`re.split`).
        :rtype: list[str]
        :returns: List of sequences and individual characters

        >>> term.split_seqs(term.underline(u'xyz'))
        ['\\x1b[4m', 'x', 'y', 'z', '\\x1b(B', '\\x1b[m']

        >>> term.split_seqs(term.underline(u'xyz'), 1)
        ['\\x1b[4m', r'xyz\\x1b(B\\x1b[m']
        """
    pattern = self._caps_unnamed_any
    result = []
    for idx, match in enumerate(re.finditer(pattern, text)):
        result.append(match.group())
        if maxsplit and idx == maxsplit:
            remaining = text[match.end():]
            if remaining:
                result[-1] += remaining
            break
    return result