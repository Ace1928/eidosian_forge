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
def clarke_thermalcond(self, structure: Structure) -> float:
    """Calculates Clarke's thermal conductivity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Clarke's thermal conductivity (in SI units)
        """
    n_sites = len(structure)
    tot_mass = sum((spec.atomic_mass for spec in structure.species))
    n_atoms = structure.composition.num_atoms
    weight = float(structure.composition.weight)
    avg_mass = 1.6605e-27 * tot_mass / n_atoms
    mass_density = 1660.5 * n_sites * weight / (n_atoms * structure.volume)
    return 0.87 * 1.3806e-23 * avg_mass ** (-2 / 3) * mass_density ** (1 / 6) * self.y_mod ** 0.5