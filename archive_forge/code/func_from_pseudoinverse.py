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
@classmethod
def from_pseudoinverse(cls, strains, stresses) -> Self:
    """
        Class method to fit an elastic tensor from stress/strain
        data. Method uses Moore-Penrose pseudo-inverse to invert
        the s = C*e equation with elastic tensor, stress, and
        strain in voigt notation.

        Args:
            stresses (Nx3x3 array-like): list or array of stresses
            strains (Nx3x3 array-like): list or array of strains
        """
    warnings.warn('Pseudo-inverse fitting of Strain/Stress lists may yield questionable results from vasp data, use with caution.')
    stresses = np.array([Stress(stress).voigt for stress in stresses])
    with warnings.catch_warnings():
        strains = np.array([Strain(strain).voigt for strain in strains])
    voigt_fit = np.transpose(np.dot(np.linalg.pinv(strains), stresses))
    return cls.from_voigt(voigt_fit)