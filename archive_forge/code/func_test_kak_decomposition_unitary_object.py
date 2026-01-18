import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_decomposition_unitary_object():
    op = cirq.ISWAP(*cirq.LineQubit.range(2)) ** 0.5
    kak = cirq.kak_decomposition(op)
    np.testing.assert_allclose(cirq.unitary(kak), cirq.unitary(op), atol=1e-08)
    assert cirq.kak_decomposition(kak) is kak