import cirq
from cirq.ops.named_qubit import _pad_digits
def test_named_qubit_str():
    q = cirq.NamedQubit('a')
    assert q.name == 'a'
    assert str(q) == 'a'
    qid = cirq.NamedQid('a', dimension=3)
    assert qid.name == 'a'
    assert str(qid) == 'a (d=3)'