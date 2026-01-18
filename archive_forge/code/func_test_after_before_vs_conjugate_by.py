import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_after_before_vs_conjugate_by():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    assert p.before(cirq.S(b)) == p.conjugated_by(cirq.S(b))
    assert p.after(cirq.S(b) ** (-1)) == p.conjugated_by(cirq.S(b))
    assert p.before(cirq.CNOT(a, b)) == p.conjugated_by(cirq.CNOT(a, b)) == p.after(cirq.CNOT(a, b))