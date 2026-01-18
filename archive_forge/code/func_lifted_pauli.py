from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def lifted_pauli(pauli_sum: Union[PauliSum, PauliTerm], qubits: List[int]) -> np.ndarray:
    """
    Takes a PauliSum object along with a list of
    qubits and returns a matrix corresponding the tensor representation of the
    object.

    Useful for generating the full Hamiltonian after a particular fermion to
    pauli transformation. For example:

    Converting a PauliSum X0Y1 + Y1X0 into the matrix

    .. code-block:: python

       [[ 0.+0.j,  0.+0.j,  0.+0.j,  0.-2.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
        [ 0.+2.j,  0.+0.j,  0.+0.j,  0.+0.j]]

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by performing the kronecker product from right to left.

    :param pauli_sum: Pauli representation of an operator
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    :return: matrix representation of the pauli_sum operator
    """
    if isinstance(pauli_sum, PauliTerm):
        pauli_sum = PauliSum([pauli_sum])
    n_qubits = len(qubits)
    result_hilbert = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)
    for term in pauli_sum.terms:
        term_hilbert = np.array([1])
        for qubit in qubits:
            term_hilbert = np.kron(QUANTUM_GATES[term[qubit]], term_hilbert)
        result_hilbert += term_hilbert * cast(complex, term.coefficient)
    return result_hilbert