import numpy as np
import pytest
import cirq
def test_measure_paulistring_terms():
    q = cirq.LineQubit.range(3)
    ps = cirq.X(q[0]) * cirq.Y(q[1]) * cirq.Z(q[2])
    assert cirq.measure_paulistring_terms(ps) == [cirq.PauliMeasurementGate([cirq.X], key=str(q[0])).on(q[0]), cirq.PauliMeasurementGate([cirq.Y], key=str(q[1])).on(q[1]), cirq.PauliMeasurementGate([cirq.Z], key=str(q[2])).on(q[2])]
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_paulistring_terms(cirq.I(q[0]) * cirq.I(q[1]))
    with pytest.raises(ValueError, match='should be an instance of cirq.PauliString'):
        _ = cirq.measure_paulistring_terms(q)