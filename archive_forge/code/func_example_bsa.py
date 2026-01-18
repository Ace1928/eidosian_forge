import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
@pytest.fixture
def example_bsa() -> 'cw.BitstringAccumulator':
    """Test fixture to create an (empty) example BitstringAccumulator"""
    q0, q1 = cirq.LineQubit.range(2)
    setting = cw.InitObsSetting(init_state=cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1), observable=cirq.X(q0) * cirq.Y(q1))
    meas_spec = _MeasurementSpec(max_setting=setting, circuit_params={'beta': 0.123, 'gamma': 0.456})
    bsa = cw.BitstringAccumulator(meas_spec=meas_spec, simul_settings=[setting, cw.InitObsSetting(init_state=setting.init_state, observable=cirq.X(q0)), cw.InitObsSetting(init_state=setting.init_state, observable=cirq.Y(q1))], qubit_to_index={q0: 0, q1: 1})
    return bsa