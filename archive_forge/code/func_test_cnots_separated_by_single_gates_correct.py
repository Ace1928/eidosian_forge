from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_cnots_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(cirq.Circuit(cirq.CNOT(a, b), cirq.H(b), cirq.CNOT(a, b)))