from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_czs_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(cirq.Circuit(cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(b), cirq.CZ(a, b)))