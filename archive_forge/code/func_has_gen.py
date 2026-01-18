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
def has_gen(obj):
    """Returns ``True`` if an operator has a generator defined."""
    if isinstance(obj, Operator):
        return obj.has_generator
    try:
        obj.generator()
    except (AttributeError, OperatorPropertyUndefined, GeneratorUndefinedError):
        return False
    return True