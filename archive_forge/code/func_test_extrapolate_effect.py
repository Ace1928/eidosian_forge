import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_extrapolate_effect():
    op1 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.5)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=1.5)
    op3 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.125)
    assert op1 ** 3 == op2
    assert op1 ** 0.25 == op3