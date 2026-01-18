import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
def compute_grouping(self, grouping_type='qwc', method='rlf'):
    """
        Compute groups of indices corresponding to commuting observables of this
        Hamiltonian, and store it in the ``grouping_indices`` attribute.

        Args:
            grouping_type (str): The type of binary relation between Pauli words used to compute the grouping.
                Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
            method (str): The graph coloring heuristic to use in solving minimum clique cover for grouping, which
                can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First).
        """
    with qml.QueuingManager.stop_recording():
        self._grouping_indices = _compute_grouping_indices(self.ops, grouping_type=grouping_type, method=method)