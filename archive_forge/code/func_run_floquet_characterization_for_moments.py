import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def run_floquet_characterization_for_moments(circuit: cirq.Circuit, sampler: Union[AbstractEngine, cirq.Sampler], processor_id: Optional[str]=None, options: FloquetPhasedFSimCalibrationOptions=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, merge_subsets: bool=True, max_layers_per_request: int=1, progress_func: Optional[Callable[[int, int], None]]=None, permit_mixed_moments: bool=False) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Extracts moments within a circuit to characterize and characterizes them against engine.

    The method calls prepare_floquet_characterization_for_moments to extract moments to characterize
    and run_calibrations to characterize them.

    Args:
        circuit: Circuit to characterize.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.ProcessorSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        max_layers_per_request: Maximum number of calibration requests issued to cirq.Engine at a
            single time. Defaults to 1.
        progress_func: Optional callback function that might be used to report the calibration
            progress. The callback is called with two integers, the first one being a number of
            layers already calibrated and the second one the total number of layers to calibrate.
        permit_mixed_moments: Whether to allow mixing single-qubit and two-qubit gates in a single
            moment.

    Returns:
        Tuple of:
          - Circuit and its mapping from moments to indices into the list of characterized requests
            (the second returned value).
          - List of PhasedFSimCalibrationRequest for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    circuit_calibration, requests = prepare_floquet_characterization_for_moments(circuit, options, gates_translator, merge_subsets=merge_subsets, permit_mixed_moments=permit_mixed_moments)
    results = run_calibrations(requests, sampler, processor_id, max_layers_per_request=max_layers_per_request, progress_func=progress_func)
    return (circuit_calibration, results)