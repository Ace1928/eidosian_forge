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
def build_setting_memory_map(self, setting: ExperimentSetting) -> Dict[str, List[float]]:
    """
        Build the memory map corresponding to the state preparation and measurement specifications
        encoded in the provided ``ExperimentSetting``, taking into account the full set of qubits
        that are present in the ``Experiment`` object.

        :return: Memory map for state prep and measurement.
        """
    meas_qubits = self.get_meas_qubits()
    in_pt = PauliTerm.from_list([(op, meas_qubits.index(cast(int, q))) for q, op in setting._in_operator()])
    out_pt = PauliTerm.from_list([(op, meas_qubits.index(cast(int, q))) for q, op in setting.out_operator])
    preparation_map = pauli_term_to_preparation_memory_map(in_pt)
    measurement_map = pauli_term_to_measurement_memory_map(out_pt)
    return {**preparation_map, **measurement_map}