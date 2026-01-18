import pytest
import sympy
import cirq
from cirq.work.observable_settings import _max_weight_state, _max_weight_observable, _hashable_param
from cirq.work import InitObsSetting, observables_to_settings, _MeasurementSpec
def test_max_weight_state():
    q0, q1 = cirq.LineQubit.range(2)
    states = [cirq.KET_PLUS(q0), cirq.KET_PLUS(q1)]
    assert _max_weight_state(states) == cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)
    states = [cirq.KET_PLUS(q0), cirq.KET_PLUS(q1), cirq.KET_MINUS(q1)]
    assert _max_weight_state(states) is None