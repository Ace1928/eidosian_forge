import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def symmetry_generators(h):
    """Compute the generators :math:`\\{\\tau_1, \\ldots, \\tau_k\\}` for a Hamiltonian over the binary
    field :math:`\\mathbb{Z}_2`.

    These correspond to the generator set of the :math:`\\mathbb{Z}_2`-symmetries present
    in the Hamiltonian as given in `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`_.

    Args:
        h (Operator): Hamiltonian for which symmetries are to be generated to perform tapering

    Returns:
        list[Operator]: list of generators of symmetries, :math:`\\tau`'s, for the Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    >>> t = symmetry_generators(H)
    >>> t
    [<Hamiltonian: terms=1, wires=[0, 1]>,
     <Hamiltonian: terms=1, wires=[0, 2]>,
     <Hamiltonian: terms=1, wires=[0, 3]>]
    >>> print(t[0])
    (1.0) [Z0 Z1]
    """
    num_qubits = len(h.wires)
    ps = pauli_sentence(h)
    binary_matrix = _binary_matrix_from_pws(list(ps), num_qubits)
    rref_binary_matrix = _reduced_row_echelon(binary_matrix)
    rref_binary_matrix_red = rref_binary_matrix[~np.all(rref_binary_matrix == 0, axis=1)]
    nullspace = _kernel(rref_binary_matrix_red)
    generators = []
    pauli_map = {'00': 'I', '10': 'X', '11': 'Y', '01': 'Z'}
    for null_vector in nullspace:
        tau = {}
        for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
            x, z = op
            tau[idx] = pauli_map[f'{x}{z}']
        ham = qml.pauli.PauliSentence({qml.pauli.PauliWord(tau): 1.0})
        ham = ham.operation(h.wires) if active_new_opmath() else ham.hamiltonian(h.wires)
        generators.append(ham)
    return generators