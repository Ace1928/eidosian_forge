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
def from_diff_fit(cls, strains, stresses, eq_stress=None, tol: float=1e-10, order=3) -> Self:
    """
        Generates an elastic tensor expansion via the fitting function
        defined below in diff_fit.
        """
    c_list = diff_fit(strains, stresses, eq_stress, order, tol)
    return cls(c_list)