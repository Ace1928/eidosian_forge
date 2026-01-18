import json
import logging
import warnings
from json import JSONEncoder
from typing import (
from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
from pyquil.experiment._program import (
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit
def build_symmetrization_memory_maps(self, qubits: Sequence[int], label: str='symmetrization') -> List[Dict[str, List[float]]]:
    """
        Build a list of memory maps to be used in a program that is trying to perform readout
        symmetrization via parametric compilation. For example, if we have the following program:

            RX(symmetrization[0]) 0
            RX(symmetrization[1]) 1
            MEASURE 0 ro[0]
            MEASURE 1 ro[1]

        We can perform exhaustive readout symmetrization on our two qubits by providing the four
        following memory maps, and then appropriately flipping the resultant bitstrings:

            {'symmetrization': [0.0, 0.0]} -> XOR results with [0,0]
            {'symmetrization': [0.0, pi]}  -> XOR results with [0,1]
            {'symmetrization': [pi, 0.0]}  -> XOR results with [1,0]
            {'symmetrization': [pi, pi]}   -> XOR results with [1,1]

        :param qubits: List of qubits to symmetrize readout for.
        :param label: Name of the declared memory region. Defaults to "symmetrization".
        :return: List of memory maps that performs the desired level of symmetrization.
        """
    num_meas_registers = len(self.get_meas_qubits())
    symm_registers = self.get_meas_registers(qubits)
    if self.symmetrization == SymmetrizationLevel.NONE:
        return [{}]
    if self.symmetrization != SymmetrizationLevel.EXHAUSTIVE:
        raise ValueError('We only support exhaustive symmetrization for now.')
    import numpy as np
    import itertools
    assignments = itertools.product(np.array([0, np.pi]), repeat=len(symm_registers))
    memory_maps = []
    for a in assignments:
        zeros = np.zeros(num_meas_registers)
        for idx, r in enumerate(symm_registers):
            zeros[r] = a[idx]
        memory_maps.append({f'{label}': list(zeros)})
    return memory_maps