from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import (
def synth_permutation_basic(pattern: list[int] | np.ndarray[int]) -> QuantumCircuit:
    """Synthesize a permutation circuit for a fully-connected
    architecture using sorting.

    More precisely, if the input permutation is a cycle of length ``m``,
    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
    if the input  permutation consists of several disjoint cycles, then each cycle
    is essentially treated independently.

    Args:
        pattern: Permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        The synthesized quantum circuit.
    """
    num_qubits = len(pattern)
    qc = QuantumCircuit(num_qubits)
    swaps = _get_ordered_swap(pattern)
    for swap in swaps:
        qc.swap(swap[0], swap[1])
    return qc