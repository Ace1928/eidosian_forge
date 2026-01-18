import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_returns_output_amplitudes_from_quirk():
    assert_url_to_circuit_returns('{"cols":[["X","Z"]]}', output_amplitudes_from_quirk=[{'r': 0, 'i': 0}, {'r': 1, 'i': 0}, {'r': 0, 'i': 0}, {'r': 0, 'i': 0}])
    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns('{"cols":[["X","Z"]]}', output_amplitudes_from_quirk=[{'r': 0, 'i': 0}, {'r': 0, 'i': 0}, {'r': 0, 'i': 1}, {'r': 0, 'i': 0}])