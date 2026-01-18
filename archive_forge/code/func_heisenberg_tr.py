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
def heisenberg_tr(self, wire_order, inverse=False):
    """Heisenberg picture representation of the linear transformation carried
        out by the gate at current parameter values.

        Given a unitary quantum gate :math:`U`, we may consider its linear
        transformation in the Heisenberg picture, :math:`U^\\dagger(\\cdot) U`.

        If the gate is Gaussian, this linear transformation preserves the polynomial order
        of any observables that are polynomials in :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.
        This also means it maps :math:`\\text{span}(\\mathbf{r})` into itself:

        .. math:: U^\\dagger \\mathbf{r}_i U = \\sum_j \\tilde{U}_{ij} \\mathbf{r}_j

        For Gaussian CV gates, this method returns the transformation matrix for
        the current parameter values of the Operation. The method is not defined
        for non-Gaussian (and non-CV) gates.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
            inverse  (bool): if True, return the inverse transformation instead

        Raises:
            RuntimeError: if the specified operation is not Gaussian or is missing the `_heisenberg_rep` method

        Returns:
            array[float]: :math:`\\tilde{U}`, the Heisenberg picture representation of the linear transformation
        """
    p = [qml.math.toarray(a) for a in self.parameters]
    if inverse:
        try:
            p[0] = np.linalg.inv(p[0])
        except np.linalg.LinAlgError:
            p[0] = -p[0]
    U = self._heisenberg_rep(p)
    if U is None:
        raise RuntimeError(f'{self.name} is not a Gaussian operation, or is missing the _heisenberg_rep method.')
    return self.heisenberg_expand(U, wire_order)