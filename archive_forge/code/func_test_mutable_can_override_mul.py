import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_can_override_mul():

    class LMul:

        def __mul__(self, other):
            return 'Yay!'

    class RMul:

        def __rmul__(self, other):
            return 'Yay!'
    assert cirq.MutablePauliString() * RMul() == 'Yay!'
    assert LMul() * cirq.MutablePauliString() == 'Yay!'