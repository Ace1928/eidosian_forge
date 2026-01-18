from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers import transformer_api, transformer_primitives, merge_k_qubit_gates
def can_merge_moment(m: 'cirq.Moment'):
    return all((protocols.num_qubits(op) == 1 and protocols.has_unitary(op) and tags_to_ignore.isdisjoint(op.tags) for op in m))