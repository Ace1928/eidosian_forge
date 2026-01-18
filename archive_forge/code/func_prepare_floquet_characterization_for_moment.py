import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_floquet_characterization_for_moment(moment: cirq.Moment, options: FloquetPhasedFSimCalibrationOptions, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, canonicalize_pairs: bool=False, sort_pairs: bool=False, permit_mixed_moments: bool=False) -> Optional[FloquetPhasedFSimCalibrationRequest]:
    """Describes a given moment in terms of a Floquet characterization request.

    Args:
        moment: Moment to characterize.
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        canonicalize_pairs: Whether to sort each of the qubit pair so that the first qubit
            is always lower than the second.
        sort_pairs: Whether to sort all the qutibt pairs extracted from the moment which will
            undergo characterization.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Instance of FloquetPhasedFSimCalibrationRequest that characterizes a given moment, or None
        when it is an empty, measurement or single-qubit gates only moment.

    Raises:
        IncompatibleMomentError when a moment contains operations other than the operations matched
        by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    return prepare_characterization_for_moment(moment=moment, options=options, gates_translator=gates_translator, canonicalize_pairs=canonicalize_pairs, sort_pairs=sort_pairs, permit_mixed_moments=permit_mixed_moments)