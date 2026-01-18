from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inDegreesMinutesSeconds(self):
    """
        The value of this angle as a degrees, minutes, seconds tuple. This
        value is immutable.

        @return: This angle expressed in degrees, minutes, seconds. L{None} if
            the angle is unknown.
        @rtype: 3-C{tuple} of C{int} (or L{None})
        """
    if self._angle is None:
        return None
    degrees = abs(int(self._angle))
    fractionalDegrees = abs(self._angle - int(self._angle))
    decimalMinutes = 60 * fractionalDegrees
    minutes = int(decimalMinutes)
    fractionalMinutes = decimalMinutes - int(decimalMinutes)
    decimalSeconds = 60 * fractionalMinutes
    return (degrees, minutes, int(decimalSeconds))