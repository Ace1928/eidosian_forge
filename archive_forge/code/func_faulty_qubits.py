import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def faulty_qubits(self):
    """Return a list of faulty qubits."""
    faulty = []
    for qubit in self._qubits:
        if not self.is_qubit_operational(qubit):
            faulty.append(qubit)
    return faulty