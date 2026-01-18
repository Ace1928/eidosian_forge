from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def all_bitstrings(n_bits: int) -> np.ndarray:
    """All bitstrings in lexicographical order as a 2d np.ndarray.

    This should be the same as ``np.array(list(itertools.product([0,1], repeat=n_bits)))``
    but faster.
    """
    n_bitstrings = 2 ** n_bits
    out = np.zeros(shape=(n_bitstrings, n_bits), dtype=np.int8)
    tf = np.array([False, True])
    for i in range(n_bits):
        j = n_bits - i - 1
        out[np.tile(np.repeat(tf, 2 ** j), 2 ** i), i] = 1
    return out