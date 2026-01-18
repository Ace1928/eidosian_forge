from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
def get_perturbed_indices(self, tol: float=1e-08):
    """
        Gets indices of perturbed elements of the deformation gradient,
        i. e. those that differ from the identity.
        """
    return list(zip(*np.where(abs(self - np.eye(3)) > tol)))