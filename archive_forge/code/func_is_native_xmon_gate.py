import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def is_native_xmon_gate(gate: cirq.Gate) -> bool:
    """Check if a gate is a native xmon gate.

    Args:
        gate: Input gate.

    Returns:
        True if the gate is native to the xmon, false otherwise.
    """
    return isinstance(gate, (cirq.CZPowGate, cirq.MeasurementGate, cirq.PhasedXPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate))