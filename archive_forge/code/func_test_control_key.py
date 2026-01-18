import cirq
def test_control_key():

    class Named:

        def _control_keys_(self):
            return frozenset([cirq.MeasurementKey('key')])

    class NoImpl:

        def _control_keys_(self):
            return NotImplemented
    assert cirq.control_keys(Named()) == {cirq.MeasurementKey('key')}
    assert not cirq.control_keys(NoImpl())
    assert not cirq.control_keys(5)