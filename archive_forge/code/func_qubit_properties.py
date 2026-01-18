from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
def qubit_properties(self, qubit: Union[int, List[int]]) -> Union[QubitProperties, List[QubitProperties]]:
    """Return QubitProperties for a given qubit.

        If there are no defined or the backend doesn't support querying these
        details this method does not need to be implemented.

        Args:
            qubit: The qubit to get the
                :class:`.QubitProperties` object for. This can
                be a single integer for 1 qubit or a list of qubits and a list
                of :class:`.QubitProperties` objects will be
                returned in the same order
        Returns:
            The :class:`~.QubitProperties` object for the
            specified qubit. If a list of qubits is provided a list will be
            returned. If properties are missing for a qubit this can be
            ``None``.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                qubit properties
        """
    if self.target.qubit_properties is None:
        raise NotImplementedError
    if isinstance(qubit, int):
        return self.target.qubit_properties[qubit]
    return [self.target.qubit_properties[q] for q in qubit]