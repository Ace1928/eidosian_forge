import pytest
import numpy as np
import cirq
def test_has_mixture():
    assert cirq.has_mixture(ReturnsValidTuple())
    assert not cirq.has_mixture(ReturnsNotImplemented())
    assert cirq.has_mixture(ReturnsMixtureButNoHasMixture())
    assert cirq.has_mixture(ReturnsUnitary())
    assert not cirq.has_mixture(ReturnsNotImplementedUnitary())