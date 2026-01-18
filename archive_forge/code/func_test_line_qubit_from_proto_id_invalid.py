import pytest
import cirq
import cirq_google.api.v2 as v2
def test_line_qubit_from_proto_id_invalid():
    with pytest.raises(ValueError, match='abc'):
        _ = v2.line_qubit_from_proto_id('abc')