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
def heisenberg_pd(self, idx):
    """Partial derivative of the Heisenberg picture transform matrix.

        Computed using grad_recipe.

        Args:
            idx (int): index of the parameter with respect to which the
                partial derivative is computed.
        Returns:
            array[float]: partial derivative
        """
    recipe = self.grad_recipe[idx]
    multiplier = 0.5
    a = 1
    shift = np.pi / 2
    default_param_shift = [[multiplier, a, shift], [-multiplier, a, -shift]]
    param_shift = default_param_shift if recipe is None else recipe
    pd = None
    p = self.parameters
    original_p_idx = p[idx]
    for c, _a, s in param_shift:
        p[idx] = _a * original_p_idx + s
        U = self._heisenberg_rep(p)
        if pd is None:
            pd = c * U
        else:
            pd += c * U
    return pd