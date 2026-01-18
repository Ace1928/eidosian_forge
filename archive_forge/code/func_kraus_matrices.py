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
def kraus_matrices(self):
    """Kraus matrices of an instantiated channel
        in the computational basis.

        Returns:
            list (array): list of Kraus matrices

        ** Example**

        >>> U = qml.AmplitudeDamping(0.1, wires=1)
        >>> U.kraus_matrices()
        [array([[1., 0.], [0., 0.9486833]]),
         array([[0., 0.31622777], [0., 0.]])]
        """
    return self.compute_kraus_matrices(*self.parameters, **self.hyperparameters)