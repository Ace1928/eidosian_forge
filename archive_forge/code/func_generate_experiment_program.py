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
def generate_experiment_program(self) -> Program:
    """
        Generate a parameterized program containing the main body program along with some additions
        to support the various state preparation, measurement, and symmetrization specifications of
        this ``Experiment``.

        State preparation and measurement are achieved via ZXZXZ-decomposed single-qubit gates,
        where the angles of each ``RZ`` rotation are declared parameters that can be assigned at
        runtime. Symmetrization is achieved by putting an ``RX`` gate (also parameterized by a
        declared value) before each ``MEASURE`` operation. In addition, a ``RESET`` operation
        is prepended to the ``Program`` if the experiment has active qubit reset enabled. Finally,
        each qubit specified in the settings is measured, and the number of shots is added.

        :return: Parameterized ``Program`` that is capable of collecting statistics for every
            ``ExperimentSetting`` in this ``Experiment``.
        """
    meas_qubits = self.get_meas_qubits()
    p = Program()
    if self.reset:
        if any((isinstance(instr, (Reset, ResetQubit)) for instr in self.program)):
            raise ValueError('RESET already added to program')
        p += RESET()
    for settings in self:
        assert len(settings) == 1
        if 'X' in str(settings[0].in_state) or 'Y' in str(settings[0].in_state):
            if 'DECLARE preparation_alpha' in self.program.out():
                raise ValueError('Memory "preparation_alpha" has been declared already.')
            if 'DECLARE preparation_beta' in self.program.out():
                raise ValueError('Memory "preparation_beta" has been declared already.')
            if 'DECLARE preparation_gamma' in self.program.out():
                raise ValueError('Memory "preparation_gamma" has been declared already.')
            p += parameterized_single_qubit_state_preparation(meas_qubits)
            break
    p += self.program
    for settings in self:
        assert len(settings) == 1
        if 'X' in str(settings[0].out_operator) or 'Y' in str(settings[0].out_operator):
            if 'DECLARE measurement_alpha' in self.program.out():
                raise ValueError('Memory "measurement_alpha" has been declared already.')
            if 'DECLARE measurement_beta' in self.program.out():
                raise ValueError('Memory "measurement_beta" has been declared already.')
            if 'DECLARE measurement_gamma' in self.program.out():
                raise ValueError('Memory "measurement_gamma" has been declared already.')
            p += parameterized_single_qubit_measurement_basis(meas_qubits)
            break
    if self.symmetrization != 0:
        if 'DECLARE symmetrization' in self.program.out():
            raise ValueError('Memory "symmetrization" has been declared already.')
        p += parameterized_readout_symmetrization(meas_qubits)
    if 'DECLARE ro' in self.program.out():
        raise ValueError('Memory "ro" has already been declared for this program.')
    p += measure_qubits(meas_qubits)
    p.wrap_in_numshots_loop(self.shots)
    return p