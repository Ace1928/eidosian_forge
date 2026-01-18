from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def tensor_up(pauli_sum: Union[PauliSum, PauliTerm], qubits: List[int]) -> np.ndarray:
    """
    Takes a PauliSum object along with a list of
    qubits and returns a matrix corresponding the tensor representation of the
    object.

    This is the same as :py:func:`lifted_pauli`. Nick R originally wrote this functionality
    and really likes the name ``tensor_up``. Who can blame him?

    :param pauli_sum: Pauli representation of an operator
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    :return: matrix representation of the pauli_sum operator
    """
    return lifted_pauli(pauli_sum=pauli_sum, qubits=qubits)