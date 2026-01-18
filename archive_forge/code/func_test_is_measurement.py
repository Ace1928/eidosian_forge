import pytest
import cirq
def test_is_measurement():
    q = cirq.NamedQubit('q')
    assert cirq.is_measurement(cirq.measure(q))
    assert cirq.is_measurement(cirq.MeasurementGate(num_qubits=1, key='b'))
    assert not cirq.is_measurement(cirq.X(q))
    assert not cirq.is_measurement(cirq.X)
    assert not cirq.is_measurement(cirq.bit_flip(1))

    class NotImplementedOperation(cirq.Operation):

        def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
            raise NotImplementedError()

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)
    assert not cirq.is_measurement(NotImplementedOperation())