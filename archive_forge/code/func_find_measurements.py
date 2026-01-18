from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def find_measurements(program: cirq.AbstractCircuit) -> List[MeasureInfo]:
    """Find measurements in the given program (circuit).

    Returns:
        List of Measurement objects with named measurements in this program.

    Raises:
        NotImplementedError: If the program is of a type that is not recognized.
        ValueError: If there is a duplicate measurement key.
    """
    if not isinstance(program, cirq.AbstractCircuit):
        raise NotImplementedError(f'Unrecognized program type: {type(program)}')
    measurements: Dict[str, MeasureInfo] = {}
    for moment in program:
        for op in moment:
            if isinstance(op.gate, cirq.MeasurementGate):
                m = MeasureInfo(key=op.gate.key, qubits=_grid_qubits(op), instances=1, invert_mask=list(op.gate.full_invert_mask()), tags=list(op.tags))
                prev_m = measurements.get(m.key)
                if prev_m is None:
                    measurements[m.key] = m
                else:
                    if m.qubits != prev_m.qubits or m.invert_mask != prev_m.invert_mask or m.tags != prev_m.tags:
                        raise ValueError(f'Incompatible repeated keys: {m}, {prev_m}')
                    prev_m.instances += 1
    return list(measurements.values())