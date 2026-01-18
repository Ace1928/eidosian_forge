from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def quick_ratio(self):
    """Return an upper bound on ratio() relatively quickly.

        This isn't defined beyond that it is an upper bound on .ratio(), and
        is faster to compute.
        """
    if self.fullbcount is None:
        self.fullbcount = fullbcount = {}
        for elt in self.b:
            fullbcount[elt] = fullbcount.get(elt, 0) + 1
    fullbcount = self.fullbcount
    avail = {}
    availhas, matches = (avail.__contains__, 0)
    for elt in self.a:
        if availhas(elt):
            numb = avail[elt]
        else:
            numb = fullbcount.get(elt, 0)
        avail[elt] = numb - 1
        if numb > 0:
            matches = matches + 1
    return _calculate_ratio(matches, len(self.a) + len(self.b))