import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('exponent,phase_exponent', itertools.product(np.arange(-2.5, 2.75, 0.25), repeat=2))
def test_exponent_consistency(exponent, phase_exponent):
    """Verifies that instances of PhasedX gate expose consistent exponents."""
    g = cirq.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)
    assert g.exponent in [exponent, -exponent]
    assert g.phase_exponent in [cirq.value.canonicalize_half_turns(g.phase_exponent)]
    g2 = cirq.PhasedXPowGate(exponent=g.exponent, phase_exponent=g.phase_exponent)
    assert g == g2
    u = cirq.protocols.unitary(g)
    u2 = cirq.protocols.unitary(g2)
    assert np.all(u == u2)