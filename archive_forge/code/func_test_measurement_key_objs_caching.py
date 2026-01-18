import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_measurement_key_objs_caching():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    m = cirq.Moment(cirq.measure(q0, key='foo'))
    assert m._measurement_key_objs is None
    key_objs = cirq.measurement_key_objs(m)
    assert m._measurement_key_objs == key_objs
    m = m.with_operation(cirq.measure(q1, key='bar'))
    assert m._measurement_key_objs == {cirq.MeasurementKey(name='bar'), cirq.MeasurementKey(name='foo')}
    m = m.with_operations(cirq.measure(q2, key='doh'), cirq.measure(q3, key='baz'))
    assert m._measurement_key_objs == {cirq.MeasurementKey(name='bar'), cirq.MeasurementKey(name='foo'), cirq.MeasurementKey(name='doh'), cirq.MeasurementKey(name='baz')}