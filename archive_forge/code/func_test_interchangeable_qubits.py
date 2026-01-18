import itertools
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate', _all_interaction_gates(exponents=(0.1, -0.25, 0.5, 1)))
def test_interchangeable_qubits(gate):
    q0, q1 = (cirq.NamedQubit('q0'), cirq.NamedQubit('q1'))
    op0 = gate(q0, q1)
    op1 = gate(q1, q0)
    mat0 = cirq.Circuit(op0).unitary()
    mat1 = cirq.Circuit(op1).unitary()
    same = op0 == op1
    same_check = cirq.allclose_up_to_global_phase(mat0, mat1)
    assert same == same_check