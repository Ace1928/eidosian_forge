import pytest
import cirq
def test_nested_key():
    with pytest.raises(ValueError, match=': is not allowed.*use `MeasurementKey.parse_serialized'):
        _ = cirq.MeasurementKey('nested:key')
    nested_key = cirq.MeasurementKey.parse_serialized('nested:key')
    assert nested_key.name == 'key'
    assert nested_key.path == ('nested',)