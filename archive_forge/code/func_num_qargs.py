from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral
from qiskit.exceptions import QiskitError
@property
def num_qargs(self):
    """Return a tuple of the number of left and right wires"""
    return (self._num_qargs_l, self._num_qargs_r)