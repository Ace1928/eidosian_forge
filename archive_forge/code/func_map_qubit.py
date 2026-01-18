import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def map_qubit(qubit: 'cirq.Qid') -> 'cirq.Qid':
    q = cast(devices.LineQubit, qubit)
    if q.x >= len(qs):
        raise IndexError(f'Only {len(qs)} qubits specified, but the given quirk circuit used the qubit at offset {q.x}. Provide more qubits.')
    return qs[q.x]