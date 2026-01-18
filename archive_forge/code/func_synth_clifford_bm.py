from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def synth_clifford_bm(clifford: Clifford) -> QuantumCircuit:
    """Optimal CX-cost decomposition of a :class:`.Clifford` operator on 2 qubits
    or 3 qubits into a :class:`.QuantumCircuit` based on the Bravyi-Maslov method [1].

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    Raises:
        QiskitError: if Clifford is on more than 3 qubits.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    num_qubits = clifford.num_qubits
    if num_qubits > 3:
        raise QiskitError('Can only decompose up to 3-qubit Clifford circuits.')
    if num_qubits == 1:
        return _decompose_clifford_1q(clifford.tableau)
    clifford_name = str(clifford)
    inv_circuit = QuantumCircuit(num_qubits, name='inv_circ')
    cost = _cx_cost(clifford)
    while cost > 0:
        clifford, inv_circuit, cost = _reduce_cost(clifford, inv_circuit, cost)
    ret_circ = QuantumCircuit(num_qubits, name=clifford_name)
    for qubit in range(num_qubits):
        pos = [qubit, qubit + num_qubits]
        circ = _decompose_clifford_1q(clifford.tableau[pos][:, pos + [-1]])
        if len(circ) > 0:
            ret_circ.append(circ, [qubit])
    if len(inv_circuit) > 0:
        ret_circ.append(inv_circuit.inverse(), range(num_qubits))
    return ret_circ.decompose()