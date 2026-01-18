from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
@classmethod
def from_parameters(cls, a: float, b: float, c: float, alpha: float, beta: float, gamma: float, *, vesta: bool=False, pbc: PbcLike=(True, True, True)) -> Self:
    """Create a Lattice using unit cell lengths (in Angstrom) and angles (in degrees).

        Args:
            a (float): *a* lattice parameter.
            b (float): *b* lattice parameter.
            c (float): *c* lattice parameter.
            alpha (float): *alpha* angle in degrees.
            beta (float): *beta* angle in degrees.
            gamma (float): *gamma* angle in degrees.
            vesta: True if you import Cartesian coordinates from VESTA.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Lattice with the specified lattice parameters.
        """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    if vesta:
        c1 = c * cos_beta
        c2 = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        vector_a = [float(a), 0.0, 0.0]
        vector_b = [b * cos_gamma, b * sin_gamma, 0]
        vector_c = [c1, c2, math.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)]
    else:
        val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
        val = np.clip(val, -1, 1)
        gamma_star = np.arccos(val)
        vector_a = [a * sin_beta, 0.0, a * cos_beta]
        vector_b = [-b * sin_alpha * np.cos(gamma_star), b * sin_alpha * np.sin(gamma_star), b * cos_alpha]
        vector_c = [0.0, 0.0, float(c)]
    return cls([vector_a, vector_b, vector_c], pbc)