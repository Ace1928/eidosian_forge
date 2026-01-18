import pytest
import numpy as np
import cirq
def test_assert_specifies_has_unitary_if_unitary_from_decompose():

    class Bad:

        def _decompose_(self):
            return []
    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())

    class Bad2:

        def _decompose_(self):
            return [cirq.X(cirq.LineQubit(0))]
    assert cirq.has_unitary(Bad2())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad2())

    class Okay:

        def _decompose_(self):
            return [cirq.depolarize(0.5).on(cirq.LineQubit(0))]
    assert not cirq.has_unitary(Okay())
    cirq.testing.assert_specifies_has_unitary_if_unitary(Okay())