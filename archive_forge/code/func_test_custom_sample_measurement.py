import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_custom_sample_measurement(self, device):
    """Test the execution of a custom sampled measurement."""
    dev = device(2)
    _skip_test_for_braket(dev)
    if not dev.shots:
        pytest.skip('Shots must be specified in the device to compute a sampled measurement.')

    class MyMeasurement(SampleMeasurement):
        """Dummy sampled measurement."""

        def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
            return 1

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return (MyMeasurement(wires=[0]), MyMeasurement(wires=[1]))
    res = circuit()
    assert qml.math.allequal(res, [1, 1])