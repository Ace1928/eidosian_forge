import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_normal_gates():
    a = cirq.LineQubit(0)
    assert cirq.X(a).conjugated_by(cirq.H(a)) == cirq.Z(a)
    assert cirq.Y(a).conjugated_by(cirq.H(a)) == -cirq.Y(a)
    assert cirq.Z(a).conjugated_by(cirq.H(a)) == cirq.X(a)
    assert cirq.X(a).conjugated_by(cirq.S(a)) == -cirq.Y(a)
    assert cirq.Y(a).conjugated_by(cirq.S(a)) == cirq.X(a)
    assert cirq.Z(a).conjugated_by(cirq.S(a)) == cirq.Z(a)