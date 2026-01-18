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
def inkey(self, timeout=None, esc_delay=0.35):
    """
        Read and return the next keyboard event within given timeout.

        Generally, this should be used inside the :meth:`raw` context manager.

        :arg float timeout: Number of seconds to wait for a keystroke before
            returning.  When ``None`` (default), this method may block
            indefinitely.
        :arg float esc_delay: To distinguish between the keystroke of
           ``KEY_ESCAPE``, and sequences beginning with escape, the parameter
           ``esc_delay`` specifies the amount of time after receiving escape
           (``chr(27)``) to seek for the completion of an application key
           before returning a :class:`~.Keystroke` instance for
           ``KEY_ESCAPE``.
        :rtype: :class:`~.Keystroke`.
        :returns: :class:`~.Keystroke`, which may be empty (``u''``) if
           ``timeout`` is specified and keystroke is not received.

        .. note:: When used without the context manager :meth:`cbreak`, or
            :meth:`raw`, :obj:`sys.__stdin__` remains line-buffered, and this
            function will block until the return key is pressed!

        .. note:: On Windows, a 10 ms sleep is added to the key press detection loop to reduce CPU
            load. Due to the behavior of :py:func:`time.sleep` on Windows, this will actually
            result in a 15.6 ms delay when using the default `time resolution
            <https://docs.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod>`_.
            Decreasing the time resolution will reduce this to 10 ms, while increasing it, which
            is rarely done, will have a perceptable impact on the behavior.
        """
    resolve = functools.partial(resolve_sequence, mapper=self._keymap, codes=self._keycodes)
    stime = time.time()
    ucs = u''
    while self._keyboard_buf:
        ucs += self._keyboard_buf.pop()
    while self.kbhit(timeout=0):
        ucs += self.getch()
    ks = resolve(text=ucs)
    while not ks and self.kbhit(timeout=_time_left(stime, timeout)):
        ucs += self.getch()
        ks = resolve(text=ucs)
    if ks.code == self.KEY_ESCAPE:
        esctime = time.time()
        while ks.code == self.KEY_ESCAPE and ucs in self._keymap_prefixes and self.kbhit(timeout=_time_left(esctime, esc_delay)):
            ucs += self.getch()
            ks = resolve(text=ucs)
    self.ungetch(ucs[len(ks):])
    return ks