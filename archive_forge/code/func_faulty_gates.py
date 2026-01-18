import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def faulty_gates(self):
    """Return a list of faulty gates."""
    faulty = []
    for gate in self.gates:
        if not self.is_gate_operational(gate.gate, gate.qubits):
            faulty.append(gate)
    return faulty