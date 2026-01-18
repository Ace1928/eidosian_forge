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
def get_effective_ecs(self, strain, order=2):
    """
        Returns the effective elastic constants
        from the elastic tensor expansion.

        Args:
            strain (Strain or 3x3 array-like): strain condition
                under which to calculate the effective constants
            order (int): order of the ecs to be returned
        """
    ec_sum = 0
    for n, ecs in enumerate(self[order - 2:]):
        ec_sum += ecs.einsum_sequence([strain] * n) / factorial(n)
    return ec_sum