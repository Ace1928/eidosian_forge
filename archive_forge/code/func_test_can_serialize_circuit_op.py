import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_can_serialize_circuit_op():
    serializer = cg.CircuitOpSerializer()
    assert serializer.can_serialize_operation(cirq.CircuitOperation(default_circuit()))
    assert not serializer.can_serialize_operation(cirq.X(cirq.GridQubit(1, 1)))