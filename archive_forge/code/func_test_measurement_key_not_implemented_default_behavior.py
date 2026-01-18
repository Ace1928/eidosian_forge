import pytest
import cirq
@pytest.mark.parametrize('key_method', [cirq.measurement_key_name, cirq.measurement_key_obj])
def test_measurement_key_not_implemented_default_behavior(key_method):

    class ReturnsNotImplemented:

        def _measurement_key_name_(self):
            return NotImplemented

        def _measurement_key_obj_(self):
            return NotImplemented
    with pytest.raises(TypeError, match='NotImplemented'):
        key_method(ReturnsNotImplemented())
    assert key_method(ReturnsNotImplemented(), None) is None
    assert key_method(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert key_method(ReturnsNotImplemented(), 'a') == 'a'