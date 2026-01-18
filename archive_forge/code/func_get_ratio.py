from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_ratio(self, max_denominator=5, index_none=None):
    """
        find the axial ratio needed for GB generator input.

        Args:
            max_denominator (int): the maximum denominator for
                the computed ratio, default to be 5.
            index_none (int): specify the irrational axis.
                0-a, 1-b, 2-c. Only may be needed for orthorhombic system.

        Returns:
            axial ratio needed for GB generator (list of integers).
        """
    structure = self.initial_structure
    lat_type = self.lat_type
    if lat_type in ('t', 'h'):
        a, _, c = structure.lattice.lengths
        if c > a:
            frac = Fraction(c ** 2 / a ** 2).limit_denominator(max_denominator)
            ratio = [frac.numerator, frac.denominator]
        else:
            frac = Fraction(a ** 2 / c ** 2).limit_denominator(max_denominator)
            ratio = [frac.denominator, frac.numerator]
    elif lat_type == 'r':
        cos_alpha = cos(structure.lattice.alpha / 180 * np.pi)
        frac = Fraction((1 + 2 * cos_alpha) / cos_alpha).limit_denominator(max_denominator)
        ratio = [frac.numerator, frac.denominator]
    elif lat_type == 'o':
        ratio = [None] * 3
        lat = (structure.lattice.c, structure.lattice.b, structure.lattice.a)
        index = [0, 1, 2]
        if index_none is None:
            min_index = np.argmin(lat)
            index.pop(min_index)
            frac1 = Fraction(lat[index[0]] ** 2 / lat[min_index] ** 2).limit_denominator(max_denominator)
            frac2 = Fraction(lat[index[1]] ** 2 / lat[min_index] ** 2).limit_denominator(max_denominator)
            com_lcm = lcm(frac1.denominator, frac2.denominator)
            ratio[min_index] = com_lcm
            ratio[index[0]] = frac1.numerator * int(round(com_lcm / frac1.denominator))
            ratio[index[1]] = frac2.numerator * int(round(com_lcm / frac2.denominator))
        else:
            index.pop(index_none)
            if lat[index[0]] > lat[index[1]]:
                frac = Fraction(lat[index[0]] ** 2 / lat[index[1]] ** 2).limit_denominator(max_denominator)
                ratio[index[0]] = frac.numerator
                ratio[index[1]] = frac.denominator
            else:
                frac = Fraction(lat[index[1]] ** 2 / lat[index[0]] ** 2).limit_denominator(max_denominator)
                ratio[index[1]] = frac.numerator
                ratio[index[0]] = frac.denominator
    elif lat_type == 'c':
        return None
    else:
        raise RuntimeError('Lattice type not implemented.')
    return ratio