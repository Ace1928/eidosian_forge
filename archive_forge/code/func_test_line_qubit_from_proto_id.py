import pytest
import cirq
import cirq_google.api.v2 as v2
def test_line_qubit_from_proto_id():
    assert v2.line_qubit_from_proto_id('1') == cirq.LineQubit(1)
    assert v2.line_qubit_from_proto_id('10') == cirq.LineQubit(10)
    assert v2.line_qubit_from_proto_id('-1') == cirq.LineQubit(-1)