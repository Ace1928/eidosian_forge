from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def test_decomposed_has_unitary():
    assert cirq.has_unitary(DecomposableGate(True))
    assert not cirq.has_unitary(DecomposableGate(False))
    assert cirq.has_unitary(DecomposableGate(True).on(a))
    assert not cirq.has_unitary(DecomposableGate(False).on(a))
    assert cirq.has_unitary(DecomposableOperation((a, b), True))
    assert cirq.has_unitary(ExampleOperation((a,)))
    assert cirq.has_unitary(ExampleOperation((a, b)))
    assert cirq.has_unitary(ExampleComposite())
    assert cirq.has_unitary(OtherComposite())