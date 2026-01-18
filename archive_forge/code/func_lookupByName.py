from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
@classmethod
def lookupByName(cls, name):
    """
        Retrieve a constant by its name or raise a C{ValueError} if there is no
        constant associated with that name.

        @param name: A C{str} giving the name of one of the constants defined
            by C{cls}.

        @raise ValueError: If C{name} is not the name of one of the constants
            defined by C{cls}.

        @return: The L{NamedConstant} associated with C{name}.
        """
    if name in cls._enumerants:
        return getattr(cls, name)
    raise ValueError(name)