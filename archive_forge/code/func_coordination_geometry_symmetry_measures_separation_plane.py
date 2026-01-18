from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def coordination_geometry_symmetry_measures_separation_plane(self, coordination_geometry, separation_plane_algo, testing=False, tested_permutations=False, points_perfect=None):
    """
        Returns the symmetry measures of the given coordination geometry "coordination_geometry" using separation
        facets to reduce the complexity of the system. Caller to the refined 2POINTS, 3POINTS and other ...

        Args:
            coordination_geometry: The coordination geometry to be investigated

        Returns:
            The symmetry measures for the given coordination geometry for each plane and permutation investigated.
        """
    permutations = []
    permutations_symmetry_measures = []
    plane_separations = []
    algos = []
    perfect2local_maps = []
    local2perfect_maps = []
    if testing:
        separation_permutations = []
    nplanes = 0
    for npoints in range(separation_plane_algo.minimum_number_of_points, min(separation_plane_algo.maximum_number_of_points, 4) + 1):
        for points_combination in itertools.combinations(self.local_geometry.coords, npoints):
            if npoints == 2:
                if collinear(points_combination[0], points_combination[1], self.local_geometry.central_site, tolerance=0.25):
                    continue
                plane = Plane.from_3points(points_combination[0], points_combination[1], self.local_geometry.central_site)
            elif npoints == 3:
                if collinear(points_combination[0], points_combination[1], points_combination[2], tolerance=0.25):
                    continue
                plane = Plane.from_3points(points_combination[0], points_combination[1], points_combination[2])
            elif npoints > 3:
                plane = Plane.from_npoints(points_combination, best_fit='least_square_distance')
            else:
                raise ValueError('Wrong number of points to initialize separation plane')
            cgsm = self._cg_csm_separation_plane(coordination_geometry=coordination_geometry, sep_plane=separation_plane_algo, local_plane=plane, plane_separations=plane_separations, dist_tolerances=DIST_TOLERANCES, testing=testing, tested_permutations=tested_permutations, points_perfect=points_perfect)
            csm, perm, algo = (cgsm[0], cgsm[1], cgsm[2])
            if csm is not None:
                permutations_symmetry_measures.extend(csm)
                permutations.extend(perm)
                for thisperm in perm:
                    p2l = {}
                    l2p = {}
                    for i_p, pp in enumerate(thisperm):
                        p2l[i_p] = pp
                        l2p[pp] = i_p
                    perfect2local_maps.append(p2l)
                    local2perfect_maps.append(l2p)
                algos.extend(algo)
                if testing:
                    separation_permutations.extend(cgsm[3])
                nplanes += 1
        if nplanes > 0:
            break
    if nplanes == 0:
        return self.coordination_geometry_symmetry_measures_fallback_random(coordination_geometry, points_perfect=points_perfect)
    if testing:
        return (permutations_symmetry_measures, permutations, separation_permutations)
    return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)