import cirq
from cirq.ops.named_qubit import _pad_digits
def test_named_qid_range():
    qids = cirq.NamedQid.range(2, prefix='a', dimension=3)
    assert qids == [cirq.NamedQid('a0', dimension=3), cirq.NamedQid('a1', dimension=3)]
    qids = cirq.NamedQid.range(-1, 4, 2, prefix='a', dimension=3)
    assert qids == [cirq.NamedQid('a-1', dimension=3), cirq.NamedQid('a1', dimension=3), cirq.NamedQid('a3', dimension=3)]
    qids = cirq.NamedQid.range(2, prefix='a', dimension=4)
    assert qids == [cirq.NamedQid('a0', dimension=4), cirq.NamedQid('a1', dimension=4)]
    qids = cirq.NamedQid.range(-1, 4, 2, prefix='a', dimension=4)
    assert qids == [cirq.NamedQid('a-1', dimension=4), cirq.NamedQid('a1', dimension=4), cirq.NamedQid('a3', dimension=4)]