from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def test_has_unitary():
    assert not cirq.has_unitary(NoMethod())
    assert not cirq.has_unitary(ReturnsNotImplemented())
    assert cirq.has_unitary(ReturnsMatrix())
    assert cirq.has_unitary(FullyImplemented(True))
    assert not cirq.has_unitary(FullyImplemented(False))