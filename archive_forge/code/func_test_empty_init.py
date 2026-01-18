import pytest
import cirq
def test_empty_init():
    with pytest.raises(TypeError, match='required positional argument'):
        _ = cirq.MeasurementKey()
    with pytest.raises(ValueError, match='valid string'):
        _ = cirq.MeasurementKey(None)
    with pytest.raises(ValueError, match='valid string'):
        _ = cirq.MeasurementKey(4.2)
    _ = cirq.MeasurementKey('')