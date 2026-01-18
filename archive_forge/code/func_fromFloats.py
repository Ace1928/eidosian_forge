from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@classmethod
def fromFloats(cls, angleValue=None, variationValue=None):
    """
        Constructs a Heading from the float values of the angle and variation.

        @param angleValue: The angle value of this heading.
        @type angleValue: C{float}
        @param variationValue: The value of the variation of this heading.
        @type variationValue: C{float}
        @return: A L{Heading} with the given values.
        """
    variation = Angle(variationValue, Angles.VARIATION)
    return cls(angleValue, variation)