from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.random import default_rng
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.random import random_unitary
from .statevector import Statevector
from .densitymatrix import DensityMatrix
def random_statevector(dims: int | tuple, seed: int | np.random.Generator | None=None) -> Statevector:
    """Generator a random Statevector.

    The statevector is sampled from the uniform distribution. This is the measure
    induced by the Haar measure on unitary matrices.

    Args:
        dims (int or tuple): the dimensions of the state.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Statevector: the random statevector.

    Reference:
        K. Zyczkowski and H. Sommers (2001), "Induced measures in the space of mixed quantum states",
        `J. Phys. A: Math. Gen. 34 7111 <https://arxiv.org/abs/quant-ph/0012101>`__.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    dim = np.prod(dims)
    vec = rng.standard_normal(dim).astype(complex)
    vec += 1j * rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec, dims=dims)