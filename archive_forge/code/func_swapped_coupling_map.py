from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
def swapped_coupling_map(self, idx: int) -> CouplingMap:
    """Returns the coupling map after applying ``idx`` swap layers of strategy.

        Args:
            idx: The number of swap layers to apply. For idx = 0, the original coupling
                map is returned.

        Returns:
            The swapped coupling map.
        """
    permutation = self.inverse_composed_permutation(idx)
    edges = [[permutation[i], permutation[j]] for i, j in self._coupling_map.get_edges()]
    return CouplingMap(couplinglist=edges)