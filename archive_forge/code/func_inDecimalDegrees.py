from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inDecimalDegrees(self):
    """
        The value of this angle in decimal degrees. This value is immutable.

        @return: This angle expressed in decimal degrees, or L{None} if the
            angle is unknown.
        @rtype: C{float} (or L{None})
        """
    return self._angle