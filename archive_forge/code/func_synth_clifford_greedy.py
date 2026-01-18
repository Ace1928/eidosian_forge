import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def synth_clifford_greedy(clifford: Clifford) -> QuantumCircuit:
    """Decompose a :class:`.Clifford` operator into a :class:`.QuantumCircuit` based
    on the greedy Clifford compiler that is described in Appendix A of
    Bravyi, Hu, Maslov and Shaydulin [1].

    This method typically yields better CX cost compared to the Aaronson-Gottesman method.

    Note that this function only implements the greedy Clifford compiler from Appendix A
    of [1], and not the templates and symbolic Pauli gates optimizations
    that are mentioned in the same paper.

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    Raises:
        QiskitError: if symplectic Gaussian elimination fails.

    References:
        1. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """
    num_qubits = clifford.num_qubits
    circ = QuantumCircuit(num_qubits, name=str(clifford))
    qubit_list = list(range(num_qubits))
    clifford_cpy = clifford.copy()
    while len(qubit_list) > 0:
        clifford_adj = clifford_cpy.copy()
        tmp = clifford_adj.destab_x.copy()
        clifford_adj.destab_x = clifford_adj.stab_z.T
        clifford_adj.destab_z = clifford_adj.destab_z.T
        clifford_adj.stab_x = clifford_adj.stab_x.T
        clifford_adj.stab_z = tmp.T
        list_greedy_cost = []
        for qubit in qubit_list:
            pauli_x = Pauli('I' * (num_qubits - qubit - 1) + 'X' + 'I' * qubit)
            pauli_x = pauli_x.evolve(clifford_adj, frame='s')
            pauli_z = Pauli('I' * (num_qubits - qubit - 1) + 'Z' + 'I' * qubit)
            pauli_z = pauli_z.evolve(clifford_adj, frame='s')
            list_pairs = []
            pauli_count = 0
            for i in qubit_list:
                typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, i)
                list_pairs.append(typeq)
                pauli_count += 1
            cost = _compute_greedy_cost(list_pairs)
            list_greedy_cost.append([cost, qubit])
        _, min_qubit = sorted(list_greedy_cost)[0]
        pauli_x = Pauli('I' * (num_qubits - min_qubit - 1) + 'X' + 'I' * min_qubit)
        pauli_x = pauli_x.evolve(clifford_adj, frame='s')
        pauli_z = Pauli('I' * (num_qubits - min_qubit - 1) + 'Z' + 'I' * min_qubit)
        pauli_z = pauli_z.evolve(clifford_adj, frame='s')
        decouple_circ, decouple_cliff = _calc_decoupling(pauli_x, pauli_z, qubit_list, min_qubit, num_qubits, clifford_cpy)
        circ = circ.compose(decouple_circ)
        clifford_cpy = decouple_cliff.adjoint().compose(clifford_cpy)
        qubit_list.remove(min_qubit)
    for qubit in range(num_qubits):
        stab = clifford_cpy.stab_phase[qubit]
        destab = clifford_cpy.destab_phase[qubit]
        if destab and stab:
            circ.y(qubit)
        elif not destab and stab:
            circ.x(qubit)
        elif destab and (not stab):
            circ.z(qubit)
    return circ