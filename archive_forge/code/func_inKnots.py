from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inKnots(self):
    """
        Returns the speed represented by this object, expressed in knots. This
        attribute is immutable.

        @return: The speed this object represents, in knots.
        @rtype: C{float}
        """
    return self._speed / MPS_PER_KNOT