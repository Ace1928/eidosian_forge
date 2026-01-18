import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def gate_to_proto(gate: cirq.Gate, qubits: Tuple[cirq.Qid, ...], delay: int) -> operations_pb2.Operation:
    if isinstance(gate, cirq.MeasurementGate):
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, measurement=_measure_to_proto(gate, qubits))
    if isinstance(gate, cirq.XPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, exp_w=_x_to_proto(gate, qubits[0]))
    if isinstance(gate, cirq.YPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, exp_w=_y_to_proto(gate, qubits[0]))
    if isinstance(gate, cirq.PhasedXPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, exp_w=_phased_x_to_proto(gate, qubits[0]))
    if isinstance(gate, cirq.ZPowGate):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, exp_z=_z_to_proto(gate, qubits[0]))
    if isinstance(gate, cirq.CZPowGate):
        if len(qubits) != 2:
            raise ValueError('Wrong number of qubits.')
        return operations_pb2.Operation(incremental_delay_picoseconds=delay, exp_11=_cz_to_proto(gate, *qubits))
    raise ValueError(f"Don't know how to serialize this gate: {gate!r}")