from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def assert_circuits_have_same_unitary_given_final_permutation(actual: circuits.AbstractCircuit, expected: circuits.AbstractCircuit, qubit_map: Dict[ops.Qid, ops.Qid]) -> None:
    """Asserts two circuits have the same unitary up to a final permuation of qubits.

    Args:
        actual: A circuit computed by some code under test.
        expected: The circuit that should have been computed.
        qubit_map: the permutation of qubits from the beginning to the end of the circuit.

    Raises:
        ValueError: if 'qubit_map' is not a mapping from the qubits in 'actual' to themselves.
        ValueError: if 'qubit_map' does not have the same set of keys and values.
    """
    if set(qubit_map.keys()) != set(qubit_map.values()):
        raise ValueError("'qubit_map' must have the same set of keys and values.")
    if not set(qubit_map.keys()).issubset(actual.all_qubits()):
        raise ValueError("'qubit_map' must be a mapping of the qubits in the circuit 'actual' to themselves.")
    actual_cp = actual.unfreeze()
    initial_qubits, sorted_qubits = zip(*sorted(qubit_map.items(), key=lambda x: x[1]))
    inverse_permutation = [sorted_qubits.index(q) for q in initial_qubits]
    actual_cp.append(ops.QubitPermutationGate(list(inverse_permutation)).on(*sorted_qubits))
    lin_alg_utils.assert_allclose_up_to_global_phase(expected.unitary(), actual_cp.unitary(), atol=1e-08)