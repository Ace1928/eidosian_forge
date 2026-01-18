import pytest
import cirq
def test_addition_subtraction_type_error():
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) + 'dave'
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) - 'dave'
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQid(1, 3) + 'dave'
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQid(1, 3) - 'dave'
    with pytest.raises(TypeError, match='Can only add LineQids with identical dimension.'):
        _ = cirq.LineQid(5, dimension=3) + cirq.LineQid(3, dimension=4)
    with pytest.raises(TypeError, match='Can only subtract LineQids with identical dimension.'):
        _ = cirq.LineQid(5, dimension=3) - cirq.LineQid(3, dimension=4)