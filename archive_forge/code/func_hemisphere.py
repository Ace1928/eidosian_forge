from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def hemisphere(self):
    """
        Gets the hemisphere of this coordinate.

        @return: A symbolic constant representing a hemisphere (one of
            L{Angles})
        """
    if self.angleType is Angles.LATITUDE:
        if self.inDecimalDegrees < 0:
            return Directions.SOUTH
        else:
            return Directions.NORTH
    elif self.angleType is Angles.LONGITUDE:
        if self.inDecimalDegrees < 0:
            return Directions.WEST
        else:
            return Directions.EAST
    else:
        raise ValueError('unknown coordinate type (cant find hemisphere)')