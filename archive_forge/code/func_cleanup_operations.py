from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def cleanup_operations(operations: Sequence[ops.Operation]):
    circuit = circuits.Circuit(operations)
    circuit = merge_single_qubit_gates_to_phased_x_and_z(circuit)
    circuit = eject_phased_paulis(circuit)
    circuit = eject_z(circuit)
    circuit = circuits.Circuit(circuit.all_operations(), strategy=circuits.InsertStrategy.EARLIEST)
    return list(circuit.all_operations())