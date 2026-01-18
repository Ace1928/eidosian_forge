from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def observables_to_binary_matrix(observables, n_qubits=None, wire_map=None):
    """Converts a list of Pauli words to the binary vector representation and yields a row matrix
    of the binary vectors.

    The dimension of the binary vectors will be implied from the highest wire being acted on
    non-trivially by the Pauli words in observables.

    Args:
        observables (list[Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]]): the list
            of Pauli words
        n_qubits (int): number of qubits to specify dimension of binary vector representation
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values


    Returns:
        array[array[int]]: a matrix whose rows are Pauli words in binary vector representation

    **Example**

    >>> observables_to_binary_matrix([X(0) @ Y(2), Z(0) @ Z(1) @ Z(2)])
    array([[1., 1., 0., 0., 1., 0.],
           [0., 0., 0., 1., 1., 1.]])
    """
    m_cols = len(observables)
    if wire_map is None:
        all_wires = Wires.all_wires([pauli_word.wires for pauli_word in observables])
        wire_map = {i: c for c, i in enumerate(all_wires)}
    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(f'n_qubits must support the highest mapped wire index {n_qubits_min}, instead got n_qubits={n_qubits}.')
    binary_mat = np.zeros((m_cols, 2 * n_qubits))
    for i in range(m_cols):
        binary_mat[i, :] = pauli_to_binary(observables[i], n_qubits=n_qubits, wire_map=wire_map)
    return binary_mat