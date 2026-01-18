from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def create_engine_validator(max_moments: int=MAX_MOMENTS, max_repetitions: int=MAX_TOTAL_REPETITIONS, max_duration_ns: int=55000) -> VALIDATOR_TYPE:
    """Creates a Callable gate set validator with a set message size.

    This validator can be used for a validator in `cg.ValidatingSampler`
    and can also be useful in generating 'engine emulators' by using
    `cg.SimulatedLocalProcessor` with this callable as a validator.

    Args:
        max_moments: Maximum number of moments to allow.
        max_repetitions: Maximum number of parameter sweep values allowed
            when summed across all sweeps and all batches.
        max_duration_ns:  Maximum duration of the circuit, in nanoseconds.
    """

    def _validator(circuits: Sequence[cirq.AbstractCircuit], sweeps: Sequence[cirq.Sweepable], repetitions: Union[int, Sequence[int]]):
        return validate_for_engine(circuits, sweeps, repetitions, max_moments, max_repetitions)
    return _validator