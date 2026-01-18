import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_text():
    p = cirq.MutablePauliString(cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)))
    assert str(cirq.MutablePauliString()) == 'mutable I'
    assert str(p) == 'mutable X(q(0))*Y(q(1))'
    cirq.testing.assert_equivalent_repr(p)