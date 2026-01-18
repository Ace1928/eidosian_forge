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
def raise_if_unphysical(func):
    """
    Wrapper for functions or properties that should raise an error
    if tensor is unphysical.
    """

    def wrapper(self, *args, **kwargs):
        if self.k_vrh < 0 or self.g_vrh < 0:
            raise ValueError('Bulk or shear modulus is negative, property cannot be determined')
        return func(self, *args, **kwargs)
    return wrapper