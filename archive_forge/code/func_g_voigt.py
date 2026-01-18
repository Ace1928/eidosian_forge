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
@property
def g_voigt(self) -> float:
    """Returns the G_v shear modulus (in eV/A^3)."""
    return (2 * self.voigt[:3, :3].trace() - np.triu(self.voigt[:3, :3]).sum() + 3 * self.voigt[3:, 3:].trace()) / 15.0