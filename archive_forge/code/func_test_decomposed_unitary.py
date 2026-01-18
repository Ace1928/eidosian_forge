from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def test_decomposed_unitary():
    np.testing.assert_allclose(cirq.unitary(DecomposableGate(True)), m1)
    np.testing.assert_allclose(cirq.unitary(DecomposableGate(True).on(a)), m1)
    np.testing.assert_allclose(cirq.unitary(DecomposableOperation((a,), True)), m1)
    np.testing.assert_allclose(cirq.unitary(DecomposableOperation((a, b), True)), m2)
    np.testing.assert_allclose(cirq.unitary(DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(cirq.unitary(ExampleOperation((a,))), np.eye(2))
    np.testing.assert_allclose(cirq.unitary(ExampleOperation((a, b))), np.eye(4))
    assert cirq.unitary(DecomposableNoUnitary((a,)), None) is None
    np.testing.assert_allclose(cirq.unitary(ExampleComposite()), np.eye(1))
    np.testing.assert_allclose(cirq.unitary(OtherComposite()), m2)