import pytest
import numpy as np
import cirq
def test_assert_specifies_has_unitary_if_unitary_from_matrix():

    class Bad:

        def _unitary_(self):
            return np.array([[1]])
    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())