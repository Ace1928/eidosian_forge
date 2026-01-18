from typing import DefaultDict, Dict, Sequence, TYPE_CHECKING, Optional
import abc
from collections import defaultdict
from cirq import circuits, devices, ops, protocols, transformers
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.permutation import (
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
@staticmethod
def canonicalize_gates(gates: LogicalGates) -> Dict[frozenset, LogicalGates]:
    """Canonicalizes a set of gates by the qubits they act on.

        Takes a set of gates specified by ordered sequences of logical
        indices, and groups those that act on the same qubits regardless of
        order."""
    canonicalized_gates: DefaultDict[frozenset, LogicalGates] = defaultdict(dict)
    for indices, gate in gates.items():
        indices = tuple(indices)
        canonicalized_gates[frozenset(indices)][indices] = gate
    return {canonical_indices: dict(list(gates.items())) for canonical_indices, gates in canonicalized_gates.items()}