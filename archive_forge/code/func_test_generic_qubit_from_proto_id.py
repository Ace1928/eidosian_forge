import pytest
import cirq
import cirq_google.api.v2 as v2
def test_generic_qubit_from_proto_id():
    assert v2.qubit_from_proto_id('1_2') == cirq.GridQubit(1, 2)
    assert v2.qubit_from_proto_id('1') == cirq.LineQubit(1)
    assert v2.qubit_from_proto_id('a') == cirq.NamedQubit('a')
    assert v2.qubit_from_proto_id('1_2_3') == cirq.NamedQubit('1_2_3')
    assert v2.qubit_from_proto_id('a') == cirq.NamedQubit('a')
    assert v2.qubit_from_proto_id('1_b') == cirq.NamedQubit('1_b')