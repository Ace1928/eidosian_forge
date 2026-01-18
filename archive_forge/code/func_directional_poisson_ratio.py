from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
def directional_poisson_ratio(self, n: ArrayLike, m: ArrayLike, tol: float=1e-08) -> float:
    """
        Calculates the poisson ratio for a specific direction
        relative to a second, orthogonal direction.

        Args:
            n (3-d vector): principal direction
            m (3-d vector): secondary direction orthogonal to n
            tol (float): tolerance for testing of orthogonality
        """
    n, m = (get_uvec(n), get_uvec(m))
    if not np.abs(np.dot(n, m)) < tol:
        raise ValueError('n and m must be orthogonal')
    v = self.compliance_tensor.einsum_sequence([n] * 2 + [m] * 2)
    v *= -1 / self.compliance_tensor.einsum_sequence([n] * 4)
    return v