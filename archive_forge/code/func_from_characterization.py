import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
@classmethod
def from_characterization(cls, qubits: Tuple[cirq.Qid, cirq.Qid], gate_calibration: PhaseCalibratedFSimGate, parameters: PhasedFSimCharacterization, characterization_index: Optional[int]) -> 'FSimPhaseCorrections':
    """Creates an operation that compensates for zeta, chi and gamma angles of the supplied
        gate and characterization.

        Args:
        Args:
            qubits: Qubits that the gate should act on.
            gate_calibration: Original, imperfect gate that is supposed to run on the hardware
                together with phase information.
            parameters: The real parameters of the supplied gate.
            characterization_index: characterization index to use at each moment with gate.
        """
    operations = gate_calibration.with_zeta_chi_gamma_compensated(qubits, parameters)
    moment_to_calibration = [None, characterization_index, None]
    return cls(operations, moment_to_calibration)