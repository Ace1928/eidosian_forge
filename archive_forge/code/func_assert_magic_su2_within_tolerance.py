import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def assert_magic_su2_within_tolerance(mat, a, b):
    M = cirq.linalg.decompositions.MAGIC
    MT = cirq.linalg.decompositions.MAGIC_CONJ_T
    recon = cirq.linalg.combinators.dot(MT, cirq.linalg.combinators.kron(a, b), M)
    assert np.allclose(recon, mat), 'Failed to decompose within tolerance.'