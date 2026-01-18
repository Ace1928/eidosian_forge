from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def scale_out_phase(unitary1: np.ndarray, unitary2: np.ndarray) -> np.ndarray:
    """
    Returns a matrix m equal to unitary1/θ where ɑ satisfies unitary2
    = e^(iθ)·unitary1.

    :param unitary1: The unitary matrix from which the constant of
        proportionality should be scaled-out.
    :param unitary2: The reference matrix.

    :return: A matrix (same shape as the input matrices) with the
             constant of proportionality scaled-out.
    """
    rescale_value = 1.0
    goodness_value = 0.0
    for j in range(unitary1.shape[0]):
        if np.abs(unitary1[j, 0]) > goodness_value:
            goodness_value = np.abs(unitary1[j, 0])
            rescale_value = unitary2[j, 0] / unitary1[j, 0]
    return rescale_value * unitary1