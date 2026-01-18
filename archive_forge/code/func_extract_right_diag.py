import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def extract_right_diag(u: np.ndarray) -> np.ndarray:
    """Extract a diagonal unitary from a 3-CNOT two-qubit unitary.

    Returns a 2-CNOT unitary D that is diagonal, so that U @ D needs only
    two CNOT gates in case the original unitary is a 3-CNOT unitary.

    See Proposition V.2 in Minimal Universal Two-Qubit CNOT-based Circuits.
    https://arxiv.org/abs/quant-ph/0308033

    Args:
        u: three-CNOT two-qubit unitary
    Returns:
        diagonal extracted from U
    """
    t = _gamma(transformations.to_special(u).T).diagonal()
    k = np.real(t[0] + t[3] - t[1] - t[2])
    psi = np.arctan2(np.imag(np.sum(t)), k)
    f = np.exp(1j * psi)
    return np.diag([1, f, f, 1])