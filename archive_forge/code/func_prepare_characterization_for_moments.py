import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_characterization_for_moments(circuit: cirq.Circuit, options: PhasedFSimCalibrationOptions[RequestT], *, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, merge_subsets: bool=True, initial: Optional[Sequence[RequestT]]=None, permit_mixed_moments: bool=False) -> Tuple[CircuitWithCalibration, List[RequestT]]:
    """Extracts a minimal set of characterization requests necessary to characterize given circuit.

    This prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    See also prepare_characterization_for_circuits_moments that operates on a list of circuits.

    Args:
        circuit: Circuit to characterize.
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: If `True` then this method tries to merge moments into the other moments
            listed previously if they can be characterized together (they have no conflicting
            operations). Otherwise, only moments of exactly the same structure are characterized
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            the requests field of the return value of prepare_characterization_for_moments invoked
            on another circuit. This might be used to find a minimal set of moments to characterize
            across many circuits.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        circuit_with_calibration:
            The circuit and its mapping from moments to indices into the list of calibration
            requests (the second returned value).
        calibrations:
            A list of calibration requests for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    if initial is None:
        initial = []
    allocations: List[Optional[int]] = []
    calibrations = list(initial)
    pairs_map = {calibration.pairs: index for index, calibration in enumerate(calibrations)}
    for moment in circuit:
        calibration = prepare_characterization_for_moment(moment, options, gates_translator=gates_translator, canonicalize_pairs=True, sort_pairs=True, permit_mixed_moments=permit_mixed_moments)
        if calibration is not None:
            if merge_subsets:
                index = _merge_into_calibrations(calibration, calibrations, pairs_map, options)
            else:
                index = _append_into_calibrations_if_missing(calibration, calibrations, pairs_map)
            allocations.append(index)
        else:
            allocations.append(None)
    return (CircuitWithCalibration(circuit, allocations), calibrations)