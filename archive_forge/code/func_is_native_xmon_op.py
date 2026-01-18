import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def is_native_xmon_op(op: cirq.Operation) -> bool:
    """Check if the gate corresponding to an operation is a native xmon gate.

    Args:
        op: Input operation.

    Returns:
        True if the operation is native to the xmon, false otherwise.
    """
    return isinstance(op, cirq.GateOperation) and is_native_xmon_gate(op.gate)