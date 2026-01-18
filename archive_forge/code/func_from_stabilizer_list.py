from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction
@classmethod
def from_stabilizer_list(cls, stabilizers: Collection[str], allow_redundant: bool=False, allow_underconstrained: bool=False) -> StabilizerState:
    """Create a stabilizer state from the collection of stabilizers.

        Args:
            stabilizers (Collection[str]): list of stabilizer strings
            allow_redundant (bool): allow redundant stabilizers (i.e., some stabilizers
                can be products of the others)
            allow_underconstrained (bool): allow underconstrained set of stabilizers (i.e.,
                the stabilizers do not specify a unique state)

        Return:
            StabilizerState: a state stabilized by stabilizers.
        """
    from qiskit.synthesis.stabilizer import synth_circuit_from_stabilizers
    circuit = synth_circuit_from_stabilizers(stabilizers, allow_redundant=allow_redundant, allow_underconstrained=allow_underconstrained)
    return cls(circuit)