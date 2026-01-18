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
def get_diff_coeff(hvec, n=1):
    """
    Helper function to find difference coefficients of an
    derivative on an arbitrary mesh.

    Args:
        hvec (1D array-like): sampling stencil
        n (int): degree of derivative to find
    """
    hvec = np.array(hvec, dtype=np.float64)
    acc = len(hvec)
    exp = np.column_stack([np.arange(acc)] * acc)
    a = np.vstack([hvec] * acc) ** exp
    b = np.zeros(acc)
    b[n] = factorial(n)
    return np.linalg.solve(a, b)