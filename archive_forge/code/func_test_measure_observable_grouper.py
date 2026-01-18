import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
@pytest.mark.parametrize('grouper', ['greedy', group_settings_greedy, _each_in_its_own_group_grouper])
def test_measure_observable_grouper(grouper):
    circuit = cirq.Circuit(cirq.X(Q) ** 0.2)
    observables = [cirq.Z(Q), cirq.Z(cirq.NamedQubit('q2'))]
    results = measure_observables(circuit, observables, cirq.Simulator(seed=52), stopping_criteria=RepetitionsStoppingCriteria(50000), grouper=grouper)
    assert len(results) == 2, 'two observables'
    np.testing.assert_allclose(0.8, results[0].mean, atol=0.05)
    np.testing.assert_allclose(1, results[1].mean, atol=1e-09)