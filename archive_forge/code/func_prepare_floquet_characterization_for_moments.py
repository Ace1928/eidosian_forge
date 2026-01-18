import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_floquet_characterization_for_moments(circuit: cirq.Circuit, options: FloquetPhasedFSimCalibrationOptions=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, merge_subsets: bool=True, initial: Optional[Sequence[FloquetPhasedFSimCalibrationRequest]]=None, permit_mixed_moments: bool=False) -> Tuple[CircuitWithCalibration, List[FloquetPhasedFSimCalibrationRequest]]:
    """Extracts a minimal set of Floquet characterization requests necessary to characterize given
    circuit.

    This variant of prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    If merge_subsets parameter is True then the method tries to merge moments into the other moments
    listed previously if they can be characterized together (they have no conflicting operations).
    If merge_subsets is False then only moments of exactly the same structure are characterized
    together.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit to characterize.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            the requests field of the return value of make_floquet_request_for_circuit invoked on
            another circuit. This might be used to find a minimal set of moments to characterize
            across many circuits.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Tuple of:
          - Circuit and its mapping from moments to indices into the list of calibration requests
            (the second returned value).
          - List of PhasedFSimCalibrationRequest for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    return cast(Tuple[CircuitWithCalibration, List[FloquetPhasedFSimCalibrationRequest]], prepare_characterization_for_moments(circuit, options, gates_translator=gates_translator, merge_subsets=merge_subsets, initial=initial, permit_mixed_moments=permit_mixed_moments))