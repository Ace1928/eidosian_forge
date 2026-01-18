import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_custom_matrix_gate():
    a, b = cirq.LineQubit.range(2)
    assert_url_to_circuit_returns('{"cols":[["~cv0d"]],"gates":[{"id":"~cv0d","matrix":"{{0,1},{1,0}}"}]}', cirq.Circuit(cirq.MatrixGate(np.array([[0, 1], [1, 0]])).on(a)))
    assert_url_to_circuit_returns('{"cols":[["~cv0d"]],"gates":[{"id":"~cv0d","name":"test","matrix":"{{0,i},{1,0}}"}]}', cirq.Circuit(cirq.MatrixGate(np.array([[0, 1j], [1, 0]])).on(a)))
    assert_url_to_circuit_returns('{"cols":[["X"],["~2hj0"]],"gates":[{"id":"~2hj0","matrix":"{{-1,0,0,0},{0,i,0,0},{0,0,1,0},{0,0,0,-i}}"}]}', cirq.Circuit(cirq.X(a), cirq.MatrixGate(np.diag([-1, 1j, 1, -1j])).on(b, a)), output_amplitudes_from_quirk=[{'r': 0, 'i': 0}, {'r': 0, 'i': 1}, {'r': 0, 'i': 0}, {'r': 0, 'i': 0}])