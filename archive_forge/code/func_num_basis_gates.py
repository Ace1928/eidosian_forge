from __future__ import annotations
import heapq
import math
from operator import itemgetter
from typing import Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate, RZXGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.one_qubit.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope
def num_basis_gates(self, unitary: Operator | np.ndarray):
    """
        Counts the number of gates that would be emitted during re-synthesis.

        .. note::
            This method is used by :class:`.ConsolidateBlocks`.
        """
    strengths = self._strength_to_infidelity(1.0)
    weyl_decomposition = TwoQubitWeylDecomposition(unitary)
    target = [getattr(weyl_decomposition, x) for x in ('a', 'b', 'c')]
    if target[-1] < -EPSILON:
        target = [np.pi / 2 - target[0], target[1], -target[2]]
    best_sequence = self._best_decomposition(target, strengths)['sequence']
    return len(best_sequence)