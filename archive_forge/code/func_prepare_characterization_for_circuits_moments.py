import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_characterization_for_circuits_moments(circuits: List[cirq.Circuit], options: PhasedFSimCalibrationOptions[RequestT], *, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, merge_subsets: bool=True, initial: Optional[Sequence[RequestT]]=None, permit_mixed_moments: bool=False) -> Tuple[List[CircuitWithCalibration], List[RequestT]]:
    """Extracts a minimal set of characterization requests necessary to characterize given circuits.

    This prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    See also prepare_characterization_for_moments that operates on a single circuit.

    Args:
        circuits: Circuits list to characterize.
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
        circuits_with_calibration:
            The circuit and its mapping from moments to indices into the list of calibration
            requests (the second returned value). When list of circuits was passed on input, this
            will be a list of CircuitWithCalibration objects corresponding to each circuit on the
            input list.
        calibrations:
            A list of calibration requests for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    requests = list(initial) if initial is not None else []
    circuits_with_calibration = []
    for circuit in circuits:
        circuit_with_calibration, requests = prepare_characterization_for_moments(circuit, options, gates_translator=gates_translator, merge_subsets=merge_subsets, initial=requests, permit_mixed_moments=permit_mixed_moments)
        circuits_with_calibration.append(circuit_with_calibration)
    return (circuits_with_calibration, requests)