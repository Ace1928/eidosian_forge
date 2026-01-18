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
def num_sorts(self):
    """Return the number of uninterpreted sorts that contain an interpretation in the model `self`.

        >>> A = DeclareSort('A')
        >>> a, b = Consts('a b', A)
        >>> s = Solver()
        >>> s.add(a != b)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.num_sorts()
        1
        """
    return int(Z3_model_get_num_sorts(self.ctx.ref(), self.model))