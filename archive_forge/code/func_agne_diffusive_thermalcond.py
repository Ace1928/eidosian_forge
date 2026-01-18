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
@due.dcite(Doi('10.1039/C7EE03256K'), description='Minimum thermal conductivity in the context of diffusion-mediated thermal transport')
@raise_if_unphysical
def agne_diffusive_thermalcond(self, structure: Structure) -> float:
    """Calculates Agne's diffusive thermal conductivity.

        Please cite the original authors if using this method
        M. T. Agne, R. Hanus, G. J. Snyder, Energy Environ. Sci. 2018, 11, 609-616.
        DOI: https://doi.org/10.1039/C7EE03256K

        Args:
            structure: pymatgen structure object

        Returns:
            float: Agne's diffusive thermal conductivity (in SI units)
        """
    n_sites = len(structure)
    site_density = 1e+30 * n_sites / structure.volume
    return 0.76 * site_density ** (2 / 3) * 1.3806e-23 * (1 / 3 * (2 * self.trans_v(structure) + self.long_v(structure)))