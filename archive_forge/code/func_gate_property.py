import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def gate_property(self, gate: str, qubits: Union[int, Iterable[int]]=None, name: str=None) -> Union[Dict[Tuple[int, ...], Dict[str, PropertyT]], Dict[str, PropertyT], PropertyT]:
    """
        Return the property of the given gate.

        Args:
            gate: Name of the gate.
            qubits: The qubit to find the property for.
            name: Optionally used to specify which gate property to return.

        Returns:
            Gate property as a tuple of the value and the time it was measured.

        Raises:
            BackendPropertyError: If the property is not found or name is
                                  specified but qubit is not.
        """
    try:
        result = self._gates[gate]
        if qubits is not None:
            if isinstance(qubits, int):
                qubits = (qubits,)
            result = result[tuple(qubits)]
            if name:
                result = result[name]
        elif name:
            raise BackendPropertyError(f'Provide qubits to get {name} of {gate}')
    except KeyError as ex:
        raise BackendPropertyError(f'Could not find the desired property for {gate}') from ex
    return result