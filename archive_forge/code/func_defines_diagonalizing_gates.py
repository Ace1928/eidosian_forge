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
@qml.BooleanFn
def defines_diagonalizing_gates(obj):
    """Returns ``True`` if an operator defines the diagonalizing gates.

    This helper function is useful if the property is to be checked in
    a queuing context, but the resulting gates must not be queued.
    """
    return obj.has_diagonalizing_gates