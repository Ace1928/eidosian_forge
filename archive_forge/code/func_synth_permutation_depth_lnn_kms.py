from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _inverse_pattern
def synth_permutation_depth_lnn_kms(pattern: list[int] | np.ndarray[int]) -> QuantumCircuit:
    """Synthesize a permutation circuit for a linear nearest-neighbor
    architecture using the Kutin, Moulton, Smithline method.

    This is the permutation synthesis algorithm from [1], section 6.
    It synthesizes any permutation of n qubits over linear nearest-neighbor
    architecture using SWAP gates with depth at most :math:`n` and size at most
    :math:`n(n-1)/2` (where both depth and size are measured with respect to SWAPs).

    Args:
        pattern: Permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Samuel A. Kutin, David Petrie Moulton and Lawren M. Smithline.
           *Computation at a distance.*,
           `arXiv:quant-ph/0701194v1 <https://arxiv.org/abs/quant-ph/0701194>`_
    """
    cur_pattern = _inverse_pattern(pattern)
    num_qubits = len(cur_pattern)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        _create_swap_layer(qc, cur_pattern, i % 2)
    return qc