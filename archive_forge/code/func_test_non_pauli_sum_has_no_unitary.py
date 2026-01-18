import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('psum', (cirq.X(q0) + cirq.Z(q0), 2 * cirq.Z(q0) * cirq.X(q1) + cirq.Y(q2), cirq.X(q0) * cirq.Z(q1) - cirq.Z(q1) * cirq.X(q0)))
def test_non_pauli_sum_has_no_unitary(psum):
    assert isinstance(psum, cirq.PauliSum)
    assert not cirq.has_unitary(psum)
    with pytest.raises(ValueError):
        _ = cirq.unitary(psum)