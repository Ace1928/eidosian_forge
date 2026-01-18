from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def real_quick_ratio(self):
    """Return an upper bound on ratio() very quickly.

        This isn't defined beyond that it is an upper bound on .ratio(), and
        is faster to compute than either .ratio() or .quick_ratio().
        """
    la, lb = (len(self.a), len(self.b))
    return _calculate_ratio(min(la, lb), la + lb)