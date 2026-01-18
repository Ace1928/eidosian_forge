import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def make_zeta_chi_gamma_compensation_for_moments(circuit: Union[cirq.Circuit, CircuitWithCalibration], characterizations: List[PhasedFSimCalibrationResult], *, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_gate_to_fsim, merge_subsets: bool=True, permit_mixed_moments: bool=False) -> CircuitWithCalibration:
    """Compensates circuit moments against errors in zeta, chi and gamma angles.

    This method creates a new circuit with a single-qubit Z gates added in a such way so that
    zeta, chi and gamma angles discovered by characterizations are cancelled-out and set to 0.

    This function preserves a moment structure of the circuit. All single qubit gates appear on new
    moments in the final circuit.

    Args:
        circuit: Circuit to compensate or instance of CircuitWithCalibration (likely returned from
            prepare_characterization_for_moments) whose mapping argument corresponds to the results
            in the characterizations argument. If circuit is passed then the method will attempt to
            match the circuit against a given characterizations. This step is can be skipped by
            passing the pre-calculated instance of CircuitWithCalibration.
        characterizations: List of characterization results (likely returned from run_calibrations).
            This should correspond to the circuit and mapping in the circuit_with_calibration
            argument.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to allow for matching moments which are subsets of the characterized
            moments. This option is only used when instance of Circuit is passed as circuit.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Calibrated circuit together with its calibration metadata in CircuitWithCalibration object.
        The calibrated circuit has single-qubit Z gates added which compensates for the true gates
        imperfections.
        The moment to calibration mapping is updated for the new circuit so that successive
        calibrations could be applied.
    """
    if isinstance(circuit, cirq.Circuit):
        circuit_with_calibration = _match_circuit_moments_with_characterizations(circuit, characterizations, gates_translator, merge_subsets, permit_mixed_moments=permit_mixed_moments)
    else:
        circuit_with_calibration = circuit
    return _make_zeta_chi_gamma_compensation(circuit_with_calibration, characterizations, gates_translator, permit_mixed_moments=permit_mixed_moments)