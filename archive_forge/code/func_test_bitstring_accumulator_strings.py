import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator_strings(example_bsa):
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    example_bsa.consume_results(bitstrings)
    q0, q1 = cirq.LineQubit.range(2)
    settings = cw.observables_to_settings([cirq.X(q0), cirq.Y(q1), cirq.X(q0) * cirq.Y(q1)], qubits=[q0, q1])
    strings_should_be = ['+Z(q(0)) * +Z(q(1)) → X(q(0)): 0.000 +- 0.577', '+Z(q(0)) * +Z(q(1)) → Y(q(1)): 0.000 +- 0.577', '+Z(q(0)) * +Z(q(1)) → X(q(0))*Y(q(1)): 0.000 +- 0.577']
    for setting, ssb in zip(settings, strings_should_be):
        assert example_bsa.summary_string(setting) == ssb, ssb
    assert str(example_bsa) == 'Accumulator +Z(q(0)) * +Z(q(1)) → X(q(0))*Y(q(1)); 4 repetitions\n  +Z(q(0)) * +Z(q(1)) → X(q(0))*Y(q(1)): 0.000 +- 0.577\n  +Z(q(0)) * +Z(q(1)) → X(q(0)): 0.000 +- 0.577\n  +Z(q(0)) * +Z(q(1)) → Y(q(1)): 0.000 +- 0.577'