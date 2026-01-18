from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def test_decompose_and_get_unitary():
    from cirq.protocols.unitary_protocol import _strat_unitary_from_decompose
    np.testing.assert_allclose(_strat_unitary_from_decompose(DecomposableOperation((a,), True)), m1)
    np.testing.assert_allclose(_strat_unitary_from_decompose(DecomposableOperation((a, b), True)), m2)
    np.testing.assert_allclose(_strat_unitary_from_decompose(DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(_strat_unitary_from_decompose(ExampleOperation((a,))), np.eye(2))
    np.testing.assert_allclose(_strat_unitary_from_decompose(ExampleOperation((a, b))), np.eye(4))
    np.testing.assert_allclose(_strat_unitary_from_decompose(ExampleComposite()), np.eye(1))
    np.testing.assert_allclose(_strat_unitary_from_decompose(OtherComposite()), m2)