from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
@property
def possible_edges(self) -> set[tuple[int, int]]:
    """Return the qubit connections that can be generated.

        Returns:
            The qubit connections that can be accommodated by the swap strategy.
        """
    if self._possible_edges is None:
        self._possible_edges = self._build_edges()
    return self._possible_edges