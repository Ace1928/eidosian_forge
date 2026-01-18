from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def get_cirq_circuit(self) -> 'cirq.Circuit':
    return circuits.Circuit((circuits.Moment(m.keys()) for m in self.ops_by_index))