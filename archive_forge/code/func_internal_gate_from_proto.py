import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def internal_gate_from_proto(msg: v2.program_pb2.InternalGate, arg_function_language: str) -> InternalGate:
    """Extracts an InternalGate object from an InternalGate proto.

    Args:
        msg: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.

    Returns:
        The deserialized InternalGate object.

    Raises:
        ValueError: On failure to parse any of the gate arguments.
    """
    gate_args = {}
    for k, v in msg.gate_args.items():
        gate_args[k] = arg_from_proto(v, arg_function_language=arg_function_language)
    return InternalGate(gate_name=str(msg.name), gate_module=str(msg.module), num_qubits=int(msg.num_qubits), **gate_args)