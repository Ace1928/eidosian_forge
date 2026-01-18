import cirq
from cirq.ops import qubit_manager as cqi
import pytest
def test_borrow_qubits():
    q = cqi.BorrowableQubit(10)
    assert q.id == 10
    assert q.dimension == 2
    assert str(q) == '_b(10)'
    assert repr(q) == 'cirq.ops.BorrowableQubit(10)'
    q = cqi.BorrowableQubit(20, dim=4)
    assert q.id == 20
    assert q.dimension == 4
    assert str(q) == '_b(20) (d=4)'
    assert repr(q) == 'cirq.ops.BorrowableQubit(20, dim=4)'
    assert cqi.BorrowableQubit(1) < cqi.BorrowableQubit(2)