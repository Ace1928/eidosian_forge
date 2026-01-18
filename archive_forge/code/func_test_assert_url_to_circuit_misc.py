import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_misc():
    a, b = cirq.LineQubit.range(2)
    assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a)), output_amplitudes_from_quirk=[{'r': 0, 'i': 0}, {'r': 0, 'i': 0}, {'r': 1, 'i': 0}, {'r': 0, 'i': 0}])
    assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a)))
    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a)), output_amplitudes_from_quirk=[{'r': 0, 'i': 0}, {'r': 0, 'i': -1}, {'r': 0, 'i': 0}, {'r': 0, 'i': 0}])
    with pytest.raises(AssertionError, match='differs from expected circuit'):
        assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.Y(b), cirq.X(a)))