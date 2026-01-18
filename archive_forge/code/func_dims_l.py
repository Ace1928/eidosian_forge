from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral
from qiskit.exceptions import QiskitError
def dims_l(self, qargs=None):
    """Return tuple of output dimension for specified subsystems."""
    if self._dims_l:
        if qargs:
            return tuple((self._dims_l[i] for i in qargs))
        return self._dims_l
    num = self._num_qargs_l if qargs is None else len(qargs)
    return num * (2,)