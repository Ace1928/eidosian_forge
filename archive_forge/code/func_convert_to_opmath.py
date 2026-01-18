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
def convert_to_opmath(op):
    """
    Converts :class:`~pennylane.Hamiltonian` and :class:`.Tensor` instances
    into arithmetic operators. Objects of any other type are returned directly.

    Arithmetic operators include :class:`~pennylane.ops.op_math.Prod`,
    :class:`~pennylane.ops.op_math.Sum` and :class:`~pennylane.ops.op_math.SProd`.

    Args:
        op (Operator): The operator instance to convert

    Returns:
        Operator: An operator using the new arithmetic operations, if relevant
    """
    if isinstance(op, qml.Hamiltonian):
        c, ops = op.terms()
        ops = tuple((convert_to_opmath(o) for o in ops))
        return qml.dot(c, ops)
    if isinstance(op, Tensor):
        return qml.prod(*op.obs)
    return op