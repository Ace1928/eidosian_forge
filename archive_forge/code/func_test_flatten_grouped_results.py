import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_flatten_grouped_results():
    q0, q1 = cirq.LineQubit.range(2)
    settings = cw.observables_to_settings([cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.Z(q0) * cirq.Z(q1)], qubits=[q0, q1])
    grouped_settings = cw.group_settings_greedy(settings)
    bsas = []
    for max_setting, simul_settings in grouped_settings.items():
        bsa = cw.BitstringAccumulator(meas_spec=_MeasurementSpec(max_setting, {}), simul_settings=simul_settings, qubit_to_index={q0: 0, q1: 1})
        bsa.consume_results(np.array([[0, 0], [0, 0], [0, 0]], dtype=np.uint8))
        bsas.append(bsa)
    results = cw.flatten_grouped_results(bsas)
    assert len(results) == 4
    for res in results:
        assert res.mean == 1
        assert res.variance == 0
        assert res.repetitions == 3