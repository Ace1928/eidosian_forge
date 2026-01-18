from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def display_covered(self, pc: float) -> str:
    """Return a displayable total percentage, as a string.

        Note that "0" is only returned when the value is truly zero, and "100"
        is only returned when the value is truly 100.  Rounding can never
        result in either "0" or "100".

        """
    if 0 < pc < self._near0:
        pc = self._near0
    elif self._near100 < pc < 100:
        pc = self._near100
    else:
        pc = round(pc, self._precision)
    return '%.*f' % (self._precision, pc)