from typing import cast, Dict, Iterable, List, Optional, Tuple
import sympy
import cirq
from cirq_google.api.v2 import batch_pb2
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter
def run_context_to_proto(sweepable: cirq.Sweepable, repetitions: int, *, out: Optional[run_context_pb2.RunContext]=None) -> run_context_pb2.RunContext:
    """Populates a RunContext protobuf message.

    Args:
        sweepable: The sweepable to include in the run context.
        repetitions: The number of repetitions for the run context.
        out: Optional message to be populated. If not given, a new message will
            be created.

    Returns:
        Populated RunContext protobuf message.
    """
    if out is None:
        out = run_context_pb2.RunContext()
    for sweep in cirq.to_sweeps(sweepable):
        sweep_proto = out.parameter_sweeps.add()
        sweep_proto.repetitions = repetitions
        sweep_to_proto(sweep, out=sweep_proto.sweep)
    return out