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
def supports_observable(self, observable):
    """Checks if an observable is supported by this device. Raises a ValueError,
         if not a subclass or string of an Observable was passed.

        Args:
            observable (type or str): observable to be checked

        Raises:
            ValueError: if `observable` is not a :class:`~.Observable` class or string

        Returns:
            bool: ``True`` iff supplied observable is supported
        """
    if isinstance(observable, type) and issubclass(observable, Observable):
        return observable.__name__ in self.observables
    if isinstance(observable, str):
        return observable in self.observables
    raise ValueError('The given observable must either be a pennylane.Observable class or a string.')