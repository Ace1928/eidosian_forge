from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
def random_pauli_list(num_qubits: int, size: int=1, seed: int | np.random.Generator | None=None, phase: bool=True):
    """Return a random PauliList.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The length of the Pauli list (Default: 1).
        seed (int or np.random.Generator): Optional. Set a fixed seed or generator for RNG.
        phase (bool): If True the Pauli phases are randomized, otherwise the phases are fixed to 0.
                     [Default: True]

    Returns:
        PauliList: a random PauliList.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    z = rng.integers(2, size=(size, num_qubits)).astype(bool)
    x = rng.integers(2, size=(size, num_qubits)).astype(bool)
    if phase:
        _phase = rng.integers(4, size=size)
        return PauliList.from_symplectic(z, x, _phase)
    return PauliList.from_symplectic(z, x)