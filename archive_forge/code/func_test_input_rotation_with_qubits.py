import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_input_rotation_with_qubits():
    a, b, c, d, e = cirq.LineQubit.range(5)
    x, y, z, t, w = cirq.LineQubit.range(10, 15)
    op = cirq.interop.quirk.QuirkInputRotationOperation(identifier='test', register=[a, b, c], base_operation=cirq.X(d).controlled_by(e), exponent_sign=-1)
    assert op.qubits == (e, d, a, b, c)
    assert op.with_qubits(x, y, z, t, w) == cirq.interop.quirk.QuirkInputRotationOperation(identifier='test', register=[z, t, w], base_operation=cirq.X(y).controlled_by(x), exponent_sign=-1)