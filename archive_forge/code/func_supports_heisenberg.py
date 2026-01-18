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
@classproperty
def supports_heisenberg(self):
    """Whether a CV operator defines a Heisenberg representation.

        This indicates that it is Gaussian and does not block the use
        of the parameter-shift differentiation method if found between the differentiated gate
        and an observable.

        Returns:
            boolean
        """
    return CV._heisenberg_rep != self._heisenberg_rep