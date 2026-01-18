from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def compile_two_qubit_gate(self, unitary: np.ndarray) -> TwoQubitGateTabulationResult:
    """Compute single qubit gates required to compile a desired unitary.

        Given a desired unitary U, this computes the sequence of 1-local gates
        $k_j$ such that the product

        $k_{n-1} A k_{n-2} A ... k_1 A k_0$

        is close to U. Here A is the base_gate of the tabulation.

        Args:
            unitary: Unitary (U above) to compile.

        Returns:
            A TwoQubitGateTabulationResult object encoding the required local
            unitaries and resulting product above.
        """
    unitary = np.asarray(unitary)
    kak_vec = cirq.kak_vector(unitary, check_preconditions=False)
    infidelities = kak_vector_infidelity(kak_vec, self.kak_vecs, ignore_equivalent_vectors=True)
    nearest_ind = int(infidelities.argmin())
    success = infidelities[nearest_ind] < self.max_expected_infidelity
    inner_gates = np.array(self.single_qubit_gates[nearest_ind])
    if inner_gates.size == 0:
        kR, kL, actual = _outer_locals_for_unitary(unitary, self.base_gate)
        return TwoQubitGateTabulationResult(self.base_gate, unitary, (kR, kL), actual, success)
    inner_gates = vector_kron(inner_gates[..., 0, :, :], inner_gates[..., 1, :, :])
    assert inner_gates.ndim == 3
    inner_product = reduce(lambda a, b: self.base_gate @ b @ a, inner_gates, self.base_gate)
    kR, kL, actual = _outer_locals_for_unitary(unitary, inner_product)
    out = [kR]
    out.extend(self.single_qubit_gates[nearest_ind])
    out.append(kL)
    return TwoQubitGateTabulationResult(self.base_gate, unitary, tuple(out), actual, success)