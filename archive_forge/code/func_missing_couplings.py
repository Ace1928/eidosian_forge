from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
@property
def missing_couplings(self) -> set[tuple[int, int]]:
    """Return the set of couplings that cannot be reached.

        Returns:
            The couplings that cannot be reached as a set of Tuples of int. Here,
            each int corresponds to a qubit in the coupling map.
        """
    if self._missing_couplings is None:
        self._missing_couplings = set(zip(*(self.distance_matrix == -1).nonzero()))
    return self._missing_couplings