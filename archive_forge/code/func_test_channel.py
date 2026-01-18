import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_channel():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    np.testing.assert_allclose(cirq.kraus(op), cirq.kraus(op.gate))
    assert cirq.has_kraus(op)
    assert cirq.kraus(cirq.testing.SingleQubitGate()(a), None) is None
    assert not cirq.has_kraus(cirq.testing.SingleQubitGate()(a))