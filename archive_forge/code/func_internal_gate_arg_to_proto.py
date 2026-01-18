import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def internal_gate_arg_to_proto(value: InternalGate, *, out: Optional[v2.program_pb2.InternalGate]=None):
    """Writes an InternalGate object into an InternalGate proto.

    Args:
        value: The gate to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.InternalGate() if out is None else out
    msg.name = value.gate_name
    msg.module = value.gate_module
    msg.num_qubits = value.num_qubits()
    for k, v in value.gate_args.items():
        arg_to_proto(value=v, out=msg.gate_args[k])
    return msg