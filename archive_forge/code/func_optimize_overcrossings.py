import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def optimize_overcrossings(self):
    """
        Minimizes the number of crossings of a strand which crosses entirely
        above the diagram by finding the path crossing over the diagram with
        the least number of overcrossings.  It begins with the longest
        overcrossing, and continues with smaller ones until it successfully
        reduces the number of crossings. Returns number of crossings removed.

        >>> L = Link([(10, 4, 11, 3),
        ...           (7, 2, 8, 3),
        ...           (8, 0, 9, 5),
        ...           (4, 10, 5, 9),
        ...           (1, 6, 2, 7),
        ...           (11, 0, 6, 1)])
        >>> len(L)
        6
        >>> L.simplify(mode='level')
        False
        >>> L.optimize_overcrossings()
        1
        """
    from . import simplify
    return simplify.strand_pickup(self, 'over')