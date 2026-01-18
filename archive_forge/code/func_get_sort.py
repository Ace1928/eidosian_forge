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
def get_sort(self, idx):
    """Return the uninterpreted sort at position `idx` < self.num_sorts().

        >>> A = DeclareSort('A')
        >>> B = DeclareSort('B')
        >>> a1, a2 = Consts('a1 a2', A)
        >>> b1, b2 = Consts('b1 b2', B)
        >>> s = Solver()
        >>> s.add(a1 != a2, b1 != b2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.num_sorts()
        2
        >>> m.get_sort(0)
        A
        >>> m.get_sort(1)
        B
        """
    if idx >= self.num_sorts():
        raise IndexError
    return _to_sort_ref(Z3_model_get_sort(self.ctx.ref(), self.model, idx), self.ctx)