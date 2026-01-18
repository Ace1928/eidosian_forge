from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def sorts(self):
    """Return all uninterpreted sorts that have an interpretation in the model `self`.

        >>> A = DeclareSort('A')
        >>> B = DeclareSort('B')
        >>> a1, a2 = Consts('a1 a2', A)
        >>> b1, b2 = Consts('b1 b2', B)
        >>> s = Solver()
        >>> s.add(a1 != a2, b1 != b2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.sorts()
        [A, B]
        """
    return [self.get_sort(i) for i in range(self.num_sorts())]