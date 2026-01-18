import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def raise_if_exceeds(self, locs):
    """
        Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.

        This is intended to be called immediately before returning *locs* from
        ``__call__`` to inform users in case their Locator returns a huge
        number of ticks, causing Matplotlib to run out of memory.

        The "strange" name of this method dates back to when it would raise an
        exception instead of emitting a log.
        """
    if len(locs) >= self.MAXTICKS:
        _log.warning('Locator attempting to generate %s ticks ([%s, ..., %s]), which exceeds Locator.MAXTICKS (%s).', len(locs), locs[0], locs[-1], self.MAXTICKS)
    return locs