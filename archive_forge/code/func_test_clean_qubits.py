import cirq
from cirq.ops import qubit_manager as cqi
import pytest
def test_clean_qubits():
    q = cqi.CleanQubit(1)
    assert q.id == 1
    assert q.dimension == 2
    assert str(q) == '_c(1)'
    assert repr(q) == 'cirq.ops.CleanQubit(1)'
    q = cqi.CleanQubit(2, dim=3)
    assert q.id == 2
    assert q.dimension == 3
    assert str(q) == '_c(2) (d=3)'
    assert repr(q) == 'cirq.ops.CleanQubit(2, dim=3)'
    assert cqi.CleanQubit(1) < cqi.CleanQubit(2)