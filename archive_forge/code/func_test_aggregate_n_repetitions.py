import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_aggregate_n_repetitions():
    with pytest.warns(UserWarning):
        reps = _aggregate_n_repetitions({5, 6})
    assert reps == 6