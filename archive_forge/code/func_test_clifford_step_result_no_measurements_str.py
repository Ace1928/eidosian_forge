import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_step_result_no_measurements_str():
    q0 = cirq.LineQubit(0)
    result = next(cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.I(q0))))
    assert str(result) == '|0⟩'