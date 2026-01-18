import cirq
from cirq.ops.named_qubit import _pad_digits
def test_named_qubit_repr():
    q = cirq.NamedQubit('a')
    assert repr(q) == "cirq.NamedQubit('a')"
    qid = cirq.NamedQid('a', dimension=3)
    assert repr(qid) == "cirq.NamedQid('a', dimension=3)"