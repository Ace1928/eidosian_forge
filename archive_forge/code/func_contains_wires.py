import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
def contains_wires(self, wires):
    """Method to determine if Wires object contains wires in another Wires object."""
    if isinstance(wires, Wires):
        return set(wires.labels).issubset(set(self._labels))
    return False