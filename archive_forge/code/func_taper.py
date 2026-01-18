import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def taper(h, generators, paulixops, paulix_sector):
    """Transform a Hamiltonian with a Clifford operator and then taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli-X operator. The list of these
    eigenvalues is defined as the Pauli sector.

    Args:
        h (Operator): Hamiltonian as a PennyLane operator
        generators (list[Operator]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]): list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators

    Returns:
        (Operator): the tapered Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> paulixops = paulix_ops(generators, 4)
    >>> paulix_sector = [1, -1, -1]
    >>> H_tapered = taper(H, generators, paulixops, paulix_sector)
    >>> print(H_tapered)
      ((-0.321034397355757+0j)) [I0]
    + ((0.1809270275619003+0j)) [X0]
    + ((0.7959678503869626+0j)) [Z0]
    """
    ps_h = pauli_sentence(h)
    return _taper_pauli_sentence(ps_h, generators, paulixops, paulix_sector)