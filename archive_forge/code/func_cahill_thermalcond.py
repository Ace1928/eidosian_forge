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
def cahill_thermalcond(self, structure: Structure) -> float:
    """Calculate Cahill's thermal conductivity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Cahill's thermal conductivity (in SI units)
        """
    n_sites = len(structure)
    site_density = 1e+30 * n_sites / structure.volume
    return 1.3806e-23 / 2.48 * site_density ** (2 / 3) * (self.long_v(structure) + 2 * self.trans_v(structure))