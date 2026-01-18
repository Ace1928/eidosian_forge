import abc
from typing import TYPE_CHECKING, Optional, FrozenSet, Iterable
import networkx as nx
from cirq import value
@property
def qubit_set(self) -> FrozenSet['cirq.Qid']:
    """Returns the set of qubits on the device.

        Returns:
            Frozenset of qubits on device.
        """
    return self._qubits_set