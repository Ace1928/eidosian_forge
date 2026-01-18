import cirq
import cirq.contrib.acquaintance as cca
def test_circular_shift_gate_permutation():
    assert cca.CircularShiftGate(3, 4).permutation() == {0: 2, 1: 0, 2: 1}
    assert cca.CircularShiftGate(4, 0).permutation() == {0: 0, 1: 1, 2: 2, 3: 3}
    assert cca.CircularShiftGate(5, 2).permutation() == {0: 3, 1: 4, 2: 0, 3: 1, 4: 2}