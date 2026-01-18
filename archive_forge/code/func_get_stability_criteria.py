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
def get_stability_criteria(self, s, n):
    """
        Gets the stability criteria from the symmetric
        Wallace tensor from an input vector and stress
        value.

        Args:
            s (float): Stress value at which to evaluate
                the stability criteria
            n (3x1 array-like): direction of the applied
                stress
        """
    n = get_uvec(n)
    stress = s * np.outer(n, n)
    sym_wallace = self.get_symmetric_wallace_tensor(stress)
    return np.linalg.det(sym_wallace.voigt)