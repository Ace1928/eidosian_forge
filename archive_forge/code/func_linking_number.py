import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def linking_number(self):
    """
        Returns the linking number of self if self has two components;
        or the sum of the linking numbers of all pairs of components
        in general.
        """
    n = 0
    for s in self.link_components:
        tally = [0] * len(self.crossings)
        for c in s:
            for i, x in enumerate(self.crossings):
                if c[0] == x:
                    tally[i] += 1
        for i, m in enumerate(tally):
            if m == 1:
                n += self.crossings[i].sign
    n = n / 4
    return n