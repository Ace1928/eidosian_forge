import pytest
import cirq
@pytest.mark.parametrize(('key_method', 'keys'), [(cirq.measurement_key_names, {'a', 'b'}), (cirq.measurement_key_objs, {'c', 'd'})])
def test_measurement_keys(key_method, keys):

    class MeasurementKeysGate(cirq.Gate):

        def _measurement_key_names_(self):
            return frozenset(['a', 'b'])

        def _measurement_key_objs_(self):
            return frozenset([cirq.MeasurementKey('c'), cirq.MeasurementKey('d')])

        def num_qubits(self) -> int:
            return 1
    a, b = cirq.LineQubit.range(2)
    assert key_method(None) == set()
    assert key_method([]) == set()
    assert key_method(cirq.X) == set()
    assert key_method(cirq.X(a)) == set()
    assert key_method(cirq.measure(a, key='out')) == {'out'}
    assert key_method(cirq.Circuit(cirq.measure(a, key='a'), cirq.measure(b, key='2'))) == {'a', '2'}
    assert key_method(MeasurementKeysGate()) == keys
    assert key_method(MeasurementKeysGate().on(a)) == keys