import abc
import copy
import functools
import itertools
import warnings
from enum import IntEnum
from typing import List
import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, eye, kron
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .utils import pauli_eigs
from .pytrees import register_pytree
def check_wires_partial_overlap(self):
    """Tests whether any two observables in the Tensor have partially
        overlapping wires and raise a warning if they do.

        .. note::

            Fully overlapping wires, i.e., observables with
            same (sets of) wires are not reported, as the ``matrix`` method is
            well-defined and implemented for this scenario.
        """
    for o1, o2 in itertools.combinations(self.obs, r=2):
        shared = qml.wires.Wires.shared_wires([o1.wires, o2.wires])
        if shared and (shared != o1.wires or shared != o2.wires):
            return 1
    return 0