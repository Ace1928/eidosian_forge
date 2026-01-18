import pytest
import cirq
import cirq_google.api.v2 as v2
def test_grid_qubit_from_proto_id_invalid():
    with pytest.raises(ValueError, match='3_3_3'):
        _ = v2.grid_qubit_from_proto_id('3_3_3')
    with pytest.raises(ValueError, match='a_2'):
        _ = v2.grid_qubit_from_proto_id('a_2')
    with pytest.raises(ValueError, match='q1_q2'):
        v2.grid_qubit_from_proto_id('q1_q2')
    with pytest.raises(ValueError, match='q-1_q2'):
        v2.grid_qubit_from_proto_id('q-1_q2')
    with pytest.raises(ValueError, match='-1_q2'):
        v2.grid_qubit_from_proto_id('-1_q2')