from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def do_gate(self, gate: Gate) -> 'NumpyWavefunctionSimulator':
    """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """
    gate_matrix, qubit_inds = _get_gate_tensor_and_qubits(gate=gate)
    self.wf = targeted_tensordot(gate=gate_matrix, wf=self.wf, wf_target_inds=qubit_inds)
    return self