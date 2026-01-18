import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator_errors():
    q0, q1 = cirq.LineQubit.range(2)
    settings = cw.observables_to_settings([cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.Z(q0) * cirq.Z(q1)], qubits=[q0, q1])
    grouped_settings = cw.group_settings_greedy(settings)
    max_setting = list(grouped_settings.keys())[0]
    simul_settings = grouped_settings[max_setting]
    with pytest.raises(ValueError):
        bsa = cw.BitstringAccumulator(meas_spec=_MeasurementSpec(max_setting, {}), simul_settings=simul_settings, qubit_to_index={q0: 0, q1: 1}, bitstrings=np.array([[0, 1], [0, 1]]), chunksizes=np.array([2]))
    with pytest.raises(ValueError):
        bsa = cw.BitstringAccumulator(meas_spec=_MeasurementSpec(max_setting, {}), simul_settings=simul_settings, qubit_to_index={q0: 0, q1: 1}, bitstrings=np.array([[0, 1], [0, 1]]), chunksizes=np.array([3]), timestamps=[datetime.datetime.now()])
    bsa = cw.BitstringAccumulator(meas_spec=_MeasurementSpec(max_setting, {}), simul_settings=simul_settings[:1], qubit_to_index={q0: 0, q1: 1})
    with pytest.raises(ValueError):
        bsa.covariance()
    with pytest.raises(ValueError):
        bsa.variance(simul_settings[0])
    with pytest.raises(ValueError):
        bsa.mean(simul_settings[0])
    bsa.consume_results(np.array([[0, 0]], dtype=np.uint8))
    assert bsa.covariance().shape == (1, 1)