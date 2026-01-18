from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
def inverse_composed_permutation(self, idx: int) -> list[int]:
    """
        Returns the inversed composed permutation of all swap layers applied up to layer
        ``idx``. Permutations are represented by list of integers where the ith element
        corresponds to the mapping of i under the permutation.

        Args:
            idx: The number of swap layers to apply.

        Returns:
            The inversed permutation as a list of integer values.
        """
    if idx not in self._inverse_composed_permutation:
        self._inverse_composed_permutation[idx] = self.apply_swap_layer(self.inverse_composed_permutation(idx - 1), idx - 1)
    return self._inverse_composed_permutation[idx]