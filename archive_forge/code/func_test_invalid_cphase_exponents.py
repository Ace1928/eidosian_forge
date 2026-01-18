from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi', itertools.product((-2.3 * np.pi, -np.pi / 7, np.pi / 5, 1.8 * np.pi), (-1.7 * np.pi, -np.pi / 5, np.pi / 7, 2.5 * np.pi)))
def test_invalid_cphase_exponents(theta, phi):
    fsim_gate = cirq.FSimGate(theta=theta, phi=phi)
    valid_exponent_intervals = cirq.compute_cphase_exponents_for_fsim_decomposition(fsim_gate)
    invalid_exponent_intervals = complement_intervals(valid_exponent_intervals)
    assert invalid_exponent_intervals
    for min_exponent, max_exponent in invalid_exponent_intervals:
        margin = 1e-08
        min_exponent += margin
        max_exponent -= margin
        assert min_exponent < max_exponent
        for exponent in np.linspace(min_exponent, max_exponent, 3):
            with pytest.raises(ValueError):
                cphase_gate = cirq.CZPowGate(exponent=exponent)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)