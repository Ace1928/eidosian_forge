import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_linear_permutation_gate_pow_not_implemented():
    permutation_gate = cca.LinearPermutationGate(3, {0: 1, 1: 2, 2: 0})
    assert permutation_gate.__pow__(0) is NotImplemented
    assert permutation_gate.__pow__(2) is NotImplemented
    assert permutation_gate.__pow__(-2) is NotImplemented
    assert permutation_gate.__pow__(0.5) is NotImplemented
    assert permutation_gate.__pow__(-0.5) is NotImplemented