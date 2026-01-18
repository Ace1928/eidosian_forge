import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pytest
import cirq
from cirq.devices import GridQubit
from cirq.vis import state_histogram
def test_get_state_histogram_multi_1():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(cirq.X.on_each(*qubits[1:]), cirq.measure(*qubits))
    r = cirq.sample(c, repetitions=5)
    values_to_plot = state_histogram.get_state_histogram(r)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(values_to_plot, expected_values)