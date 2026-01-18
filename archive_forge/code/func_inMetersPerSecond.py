from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inMetersPerSecond(self):
    """
        The speed that this object represents, expressed in meters per second.
        This attribute is immutable.

        @return: The speed this object represents, in meters per second.
        @rtype: C{float}
        """
    return self._speed