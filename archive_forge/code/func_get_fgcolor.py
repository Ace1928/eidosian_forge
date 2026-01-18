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
def get_fgcolor(self, timeout=None):
    """
        Return tuple (r, g, b) of foreground color.

        :arg float timeout: Return after time elapsed in seconds with value ``(-1, -1, -1)``
            indicating that the remote end did not respond.
        :rtype: tuple
        :returns: foreground color as tuple in form of ``(r, g, b)``.  When a timeout is specified,
            always ensure the return value is checked for ``(-1, -1, -1)``.

        The foreground color is determined by emitting an `OSC 10 color query
        <https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands>`_.
        """
    match = self._query_response(u'\x1b]10;?\x07', re.compile(u'\x1b]10;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)\x07'), timeout)
    return tuple((int(val, 16) for val in match.groups())) if match else (-1, -1, -1)