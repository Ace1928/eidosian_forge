from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def create_program_validator(max_size: int=MAX_MESSAGE_SIZE) -> PROGRAM_VALIDATOR_TYPE:
    """Creates a Callable program validator with a set message size.

    This validator can be used for a validator in `cg.ValidatingSampler`
    and can also be useful in generating 'engine emulators' by using
    `cg.SimulatedLocalProcessor` with this callable as a program_validator.

    Args:
        max_size:  proto size limit to check against.

    Returns: Callable to use in validation with the max_size already set.
    """

    def _validator(circuits: Sequence[cirq.AbstractCircuit], sweeps: Sequence[cirq.Sweepable], repetitions: int, serializer: Serializer):
        return validate_program(circuits, sweeps, repetitions, serializer, max_size)
    return _validator