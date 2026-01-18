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
@raise_if_unphysical
def debye_temperature(self, structure: Structure) -> float:
    """
        Estimates the Debye temperature from longitudinal and transverse sound velocities.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Debye temperature (in SI units)
        """
    v0 = structure.volume * 1e-30 / len(structure)
    vl, vt = (self.long_v(structure), self.trans_v(structure))
    vm = 3 ** (1 / 3) * (1 / vl ** 3 + 2 / vt ** 3) ** (-1 / 3)
    return 1.05457e-34 / 1.38065e-23 * vm * (6 * np.pi ** 2 / v0) ** (1 / 3)