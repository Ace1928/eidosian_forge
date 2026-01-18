import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator_stats_2():
    bitstrings = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], np.uint8)
    chunksizes = np.asarray([4])
    timestamps = np.asarray([datetime.datetime.now()])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    settings = list(cw.observables_to_settings([cirq.Z(a) * 5, cirq.Z(b) * 3], qubits=[a, b]))
    meas_spec = _MeasurementSpec(settings[0], {})
    bsa = cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=settings, qubit_to_index=qubit_to_index, bitstrings=bitstrings, chunksizes=chunksizes, timestamps=timestamps)
    np.testing.assert_allclose([0, 0], bsa.means())
    should_be = 4 * np.array([[5 * 5, 5 * 3], [3 * 5, 3 * 3]])
    should_be = should_be / (4 - 1)
    should_be = should_be / 4
    np.testing.assert_allclose(should_be, bsa.covariance())
    for setting, var in zip(settings, [4 * 5 ** 2, 4 * 3 ** 2]):
        np.testing.assert_allclose(0, bsa.mean(setting))
        np.testing.assert_allclose(var / 4 / (4 - 1), bsa.variance(setting))
        np.testing.assert_allclose(np.sqrt(var / 4 / (4 - 1)), bsa.stderr(setting))