import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def is_gate_operational(self, gate: str, qubits: Union[int, Iterable[int]]=None) -> bool:
    """
        Return the operational status of the given gate.

        Args:
            gate: Name of the gate.
            qubits: The qubit to find the operational status for.

        Returns:
            bool: Operational status of the given gate. True if the gate is operational,
            False otherwise.
        """
    properties = self.gate_property(gate, qubits)
    if 'operational' in properties:
        return bool(properties['operational'][0])
    return True