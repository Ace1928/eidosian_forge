from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.linear.linear_matrix_utils import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import (
def synth_stabilizer_layers(stab: StabilizerState, cz_synth_func: Callable[[np.ndarray], QuantumCircuit]=_default_cz_synth_func, cz_func_reverse_qubits: bool=False, validate: bool=False) -> QuantumCircuit:
    """Synthesis of a stabilizer state into layers.

    It provides a similar decomposition to the synthesis described in Lemma 8 of reference [1],
    without the initial Hadamard-free sub-circuit which do not affect the stabilizer state.

    For example, a 5-qubit stabilizer state is decomposed into the following layers:

    .. parsed-literal::
             ┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││     ││     ││        │
        q_2: ┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└─────┘└─────┘└────────┘

    Args:
        stab: A stabilizer state.
        cz_synth_func: A function to decompose the CZ sub-circuit.
            It gets as input a boolean symmetric matrix, and outputs a :class:`.QuantumCircuit`.
        cz_func_reverse_qubits: ``True`` only if ``cz_synth_func`` is
            :func:`.synth_cz_depth_line_mr`,
            since this function returns a circuit that reverts the order of qubits.
        validate: If ``True``, validates the synthesis process.

    Returns:
        A circuit implementation of the stabilizer state.

    Raises:
        QiskitError: if the input is not a :class:`.StabilizerState`.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    if not isinstance(stab, StabilizerState):
        raise QiskitError('The input is not a StabilizerState.')
    cliff = stab.clifford
    num_qubits = cliff.num_qubits
    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff
    H1_circ, cliff1 = _create_graph_state(cliff0, validate=validate)
    H2_circ, CZ1_circ, S1_circ, _ = _decompose_graph_state(cliff1, validate=validate, cz_synth_func=cz_synth_func)
    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)
    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)
    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list)
    from qiskit.quantum_info.operators.symplectic import Clifford
    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff_stabilizer(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)
    return layeredCircuit