import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_returns_diagram():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', diagram='0: ───X───')
    with pytest.raises(AssertionError, match='text diagram differs'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', diagram='not even close')