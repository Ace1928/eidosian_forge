import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def assert_kronecker_factorization_within_tolerance(matrix, g, f1, f2):
    restored = g * cirq.linalg.combinators.kron(f1, f2)
    assert not np.any(np.isnan(restored)), 'NaN in kronecker product.'
    assert np.allclose(restored, matrix), "Can't factor kronecker product."