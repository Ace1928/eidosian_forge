import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_returns_maps():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 1})
    assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 1, 1: 0})
    with pytest.raises(AssertionError, match='was mapped to 0b1'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 0})
    with pytest.raises(AssertionError, match='was mapped to None'):
        assert_url_to_circuit_returns('{"cols":[["H"]]}', maps={0: 0})