from __future__ import annotations
import numpy as np
from qiskit.circuit.gate import Gate
from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses
def find_basic_approximation(self, sequence: GateSequence) -> Gate:
    """Finds gate in ``self._basic_approximations`` that best represents ``sequence``.

        Args:
            sequence: The gate to find the approximation to.

        Returns:
            Gate in basic approximations that is closest to ``sequence``.
        """

    def key(x):
        return np.linalg.norm(np.subtract(x.product, sequence.product))
    best = min(self.basic_approximations, key=key)
    return best