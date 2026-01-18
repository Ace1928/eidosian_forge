from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
@classmethod
def lookupByValue(cls, value):
    """
        Retrieve a constant by its value or raise a C{ValueError} if there is
        no constant associated with that value.

        @param value: The value of one of the constants defined by C{cls}.

        @raise ValueError: If C{value} is not the value of one of the constants
            defined by C{cls}.

        @return: The L{ValueConstant} associated with C{value}.
        """
    for constant in cls.iterconstants():
        if constant.value == value:
            return constant
    raise ValueError(value)