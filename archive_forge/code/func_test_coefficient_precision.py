import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_coefficient_precision():
    qs = cirq.LineQubit.range(4 * 10 ** 3)
    r = cirq.MutablePauliString({q: cirq.X for q in qs})
    r2 = cirq.MutablePauliString({q: cirq.Y for q in qs})
    r2 *= r
    assert r2.coefficient == 1