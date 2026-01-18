import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_measure_observable_bad_grouper():
    circuit = cirq.Circuit(cirq.X(Q) ** 0.2)
    observables = [cirq.Z(Q), cirq.Z(cirq.NamedQubit('q2'))]
    with pytest.raises(ValueError, match='Unknown grouping function'):
        _ = measure_observables(circuit, observables, cirq.Simulator(seed=52), stopping_criteria=RepetitionsStoppingCriteria(50000), grouper='super fancy grouper')