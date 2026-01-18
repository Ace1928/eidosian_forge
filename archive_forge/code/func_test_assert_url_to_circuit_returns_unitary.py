import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_returns_unitary():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', unitary=cirq.unitary(cirq.X))
    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', unitary=np.eye(2))