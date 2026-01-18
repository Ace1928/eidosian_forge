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
def generate_calibration_experiment(self) -> 'Experiment':
    """
        Generate another ``Experiment`` object that can be used to calibrate the various multi-qubit
        observables involved in this ``Experiment``. This is achieved by preparing the plus-one
        (minus-one) eigenstate of each ``out_operator``, and measuring the resulting expectation
        value of the same ``out_operator``. Ideally, this would always give +1 (-1), but when
        symmetric readout error is present the effect is to scale the resultant expectations by some
        constant factor. Determining this scale factor is what we call *readout calibration*, and
        then the readout error in subsequent measurements can then be mitigated by simply dividing
        by the scale factor.

        :return: A new ``Experiment`` that can calibrate the readout error of all the
            observables involved in this experiment.
        """
    if self.calibration != CalibrationMethod.PLUS_EIGENSTATE:
        raise ValueError('We currently only support the "plus eigenstate" calibration method.')
    calibration_settings = []
    for settings in self:
        assert len(settings) == 1
        calibration_settings.append(ExperimentSetting(in_state=_pauli_to_product_state(settings[0].out_operator), out_operator=settings[0].out_operator, additional_expectations=settings[0].additional_expectations))
    calibration_program = Program()
    if self.reset:
        calibration_program += RESET()
    calibration_program.wrap_in_numshots_loop(self.shots)
    if self.symmetrization != SymmetrizationLevel.EXHAUSTIVE:
        raise ValueError('We currently only support calibration for exhaustive symmetrization')
    return Experiment(settings=calibration_settings, program=calibration_program, symmetrization=SymmetrizationLevel.EXHAUSTIVE, calibration=CalibrationMethod.NONE)