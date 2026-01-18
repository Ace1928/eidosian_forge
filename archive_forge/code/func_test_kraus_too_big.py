import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_kraus_too_big():
    m = cirq.Moment(cirq.IdentityGate(11).on(*cirq.LineQubit.range(11)))
    assert not cirq.has_kraus(m)
    assert not m._has_superoperator_()
    assert m._kraus_() is NotImplemented
    assert m._superoperator_() is NotImplemented
    assert cirq.kraus(m, default=None) is None