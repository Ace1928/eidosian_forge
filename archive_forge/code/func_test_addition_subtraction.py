import pytest
import cirq
def test_addition_subtraction():
    assert cirq.LineQubit(1) + 2 == cirq.LineQubit(3)
    assert cirq.LineQubit(3) - 1 == cirq.LineQubit(2)
    assert 1 + cirq.LineQubit(4) == cirq.LineQubit(5)
    assert 5 - cirq.LineQubit(3) == cirq.LineQubit(2)
    assert cirq.LineQid(1, 3) + 2 == cirq.LineQid(3, 3)
    assert cirq.LineQid(3, 3) - 1 == cirq.LineQid(2, 3)
    assert 1 + cirq.LineQid(4, 3) == cirq.LineQid(5, 3)
    assert 5 - cirq.LineQid(3, 3) == cirq.LineQid(2, 3)
    assert cirq.LineQid(1, dimension=3) + cirq.LineQid(3, dimension=3) == cirq.LineQid(4, dimension=3)
    assert cirq.LineQid(3, dimension=3) - cirq.LineQid(2, dimension=3) == cirq.LineQid(1, dimension=3)