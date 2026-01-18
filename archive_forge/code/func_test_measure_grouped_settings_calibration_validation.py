import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_measure_grouped_settings_calibration_validation():
    mock_ro_calib = _MockBitstringAccumulator()
    grouped_settings, qubits = _get_some_grouped_settings()
    with pytest.raises(ValueError, match='Readout calibration only works if `readout_symmetrization` is enabled'):
        cw.measure_grouped_settings(circuit=cirq.Circuit(cirq.I.on_each(*qubits)), grouped_settings=grouped_settings, sampler=cirq.Simulator(), stopping_criteria=cw.RepetitionsStoppingCriteria(10000), readout_calibrations=mock_ro_calib, readout_symmetrization=False)