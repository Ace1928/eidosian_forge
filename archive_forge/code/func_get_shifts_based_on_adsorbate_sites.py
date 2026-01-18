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
def get_shifts_based_on_adsorbate_sites(self, tolerance: float=0.1) -> list[tuple[float, float]]:
    """Computes possible in-plane shifts based on an adsorbate site  algorithm.

        Args:
            tolerance: tolerance for "uniqueness" for shifts in Cartesian unit
                This is usually Angstroms.
        """
    substrate, film = (self.substrate, self.film)
    substrate_surface_sites = np.dot(list(chain.from_iterable(AdsorbateSiteFinder(substrate).find_adsorption_sites().values())), substrate.lattice.inv_matrix)
    film_surface_sites = np.dot(list(chain.from_iterable(AdsorbateSiteFinder(film).find_adsorption_sites().values())), film.lattice.inv_matrix)
    pos_shift = np.array([np.add(np.multiply(-1, film_shift), sub_shift) for film_shift, sub_shift in product(film_surface_sites, substrate_surface_sites)])

    def _base_round(x, base=0.05):
        return base * (np.array(x) / base).round()
    pos_shift[:, 0] = _base_round(pos_shift[:, 0], base=tolerance / substrate.lattice.a)
    pos_shift[:, 1] = _base_round(pos_shift[:, 1], base=tolerance / substrate.lattice.b)
    pos_shift = pos_shift[:, 0:2]
    return list(np.unique(pos_shift, axis=0))