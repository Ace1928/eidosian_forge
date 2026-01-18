from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.linear.linear_matrix_utils import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import (
def synth_stabilizer_depth_lnn(stab: StabilizerState) -> QuantumCircuit:
    """Synthesis of an n-qubit stabilizer state for linear-nearest neighbour connectivity,
    in 2-qubit depth :math:`2n+2` and two distinct CX layers, using :class:`.CXGate`\\ s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        stab: A stabilizer state.

    Returns:
        A circuit implementation of the stabilizer state.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
        2. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """
    circ = synth_stabilizer_layers(stab, cz_synth_func=synth_cz_depth_line_mr, cz_func_reverse_qubits=True)
    return circ