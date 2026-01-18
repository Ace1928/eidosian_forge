import pytest
import numpy as np
import cirq
def test_assert_specifies_has_unitary_if_unitary_from_apply():

    class Bad(cirq.Operation):

        @property
        def qubits(self):
            return ()

        def with_qubits(self, *new_qubits):
            return self

        def _apply_unitary_(self, args):
            return args.target_tensor
    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())