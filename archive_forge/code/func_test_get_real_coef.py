import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_get_real_coef():
    q0 = cirq.LineQubit(0)
    assert _check_and_get_real_coef(cirq.Z(q0) * 2, atol=1e-08) == 2
    assert _check_and_get_real_coef(cirq.Z(q0) * complex(2.0), atol=1e-08) == 2
    with pytest.raises(ValueError):
        _check_and_get_real_coef(cirq.Z(q0) * 2j, atol=1e-08)