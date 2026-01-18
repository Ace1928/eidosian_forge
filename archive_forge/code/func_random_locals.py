import itertools
import numpy as np
import pytest
import cirq
import sympy
def random_locals(x, y, z, seed=None):
    rng = np.random.RandomState(seed)
    a0 = cirq.testing.random_unitary(2, random_state=rng)
    a1 = cirq.testing.random_unitary(2, random_state=rng)
    b0 = cirq.testing.random_unitary(2, random_state=rng)
    b1 = cirq.testing.random_unitary(2, random_state=rng)
    return cirq.unitary(cirq.KakDecomposition(interaction_coefficients=(x, y, z), single_qubit_operations_before=(a0, a1), single_qubit_operations_after=(b0, b1)))