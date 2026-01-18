from typing import Tuple
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
@staticmethod
def summands_commute(operator: SparsePauliOp) -> bool:
    """Check if all summands in the evolved operator commute.

        Args:
            operator: The operator to check if all its summands commute.

        Returns:
            True if all summands commute, False otherwise.
        """
    commuting_subparts = operator.paulis.group_qubit_wise_commuting()
    return len(commuting_subparts) == 1