import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_custom_state_measurement(self, device):
    """Test the execution of a custom state measurement."""
    dev = device(2)
    _skip_test_for_braket(dev)
    if dev.shots:
        pytest.skip("Some plugins don't update state information when shots is not None.")

    class MyMeasurement(StateMeasurement):
        """Dummy state measurement."""

        def process_state(self, state, wire_order):
            return 1

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return MyMeasurement()
    assert circuit() == 1