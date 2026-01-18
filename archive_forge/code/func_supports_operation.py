import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager
def supports_operation(self, operation):
    """Checks if an operation is supported by this device.

        Args:
            operation (type or str): operation to be checked

        Raises:
            ValueError: if `operation` is not a :class:`~.Operation` class or string

        Returns:
            bool: ``True`` if supplied operation is supported
        """
    if isinstance(operation, type) and issubclass(operation, Operation):
        return operation.__name__ in self.operations
    if isinstance(operation, str):
        return operation in self.operations
    raise ValueError('The given operation must either be a pennylane.Operation class or a string.')