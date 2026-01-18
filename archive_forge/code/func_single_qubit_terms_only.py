from typing import Tuple
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
@staticmethod
def single_qubit_terms_only(operator: SparsePauliOp) -> bool:
    """Determine if the Paulis are made of single qubit terms only.

        Args:
            operator: The operator to check if it consists only of single qubit terms.

        Returns:
            True if the operator consists of only single qubit terms (like ``IIX + IZI``),
            and False otherwise.
        """
    for pauli in operator.paulis:
        if sum(np.logical_or(pauli.x, pauli.z)) > 1:
            return False
    return True