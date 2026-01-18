import numpy as np
import pytest
import cirq
def test_assert_pauli_expansion_is_consistent_with_unitary():
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(GoodGateExplicitPauliExpansion())
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(GoodGateNoPauliExpansion())
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(GoodGateNoUnitary())
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(GoodGateNoPauliExpansionNoUnitary())
    with pytest.raises(AssertionError):
        cirq.testing.assert_pauli_expansion_is_consistent_with_unitary(BadGateInconsistentPauliExpansion())