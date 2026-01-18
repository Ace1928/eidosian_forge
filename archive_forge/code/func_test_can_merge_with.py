import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_can_merge_with():
    q0, q1 = _make_qubits(2)
    op1 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.75)
    assert op1.can_merge_with(op2)
    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.75)
    assert op1.can_merge_with(op2)
    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y}, -1), exponent_neg=0.75)
    assert not op1.can_merge_with(op2)
    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), qubits=[q0, q1], exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.75)
    assert not op1.can_merge_with(op2)