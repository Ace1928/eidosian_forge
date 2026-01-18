from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
def swap_layer(self, idx: int) -> list[tuple[int, int]]:
    """Return the layer of swaps at the given index.

        Args:
            idx: The index of the returned swap layer.

        Returns:
            A copy of the swap layer at ``idx`` to avoid any unintentional modification to
            the swap strategy.
        """
    return list(self._swap_layers[idx])