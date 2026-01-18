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
def coordination_geometry_symmetry_measures_sepplane_optim(self, coordination_geometry, points_perfect=None, nb_set=None, optimization=None):
    """Returns the symmetry measures of a given coordination_geometry for a set of
        permutations depending on the permutation setup. Depending on the parameters of
        the LocalGeometryFinder and on the coordination geometry, different methods are called.

        Args:
            coordination_geometry: Coordination geometry for which the symmetry measures are looked for

        Raises:
            NotImplementedError: if the permutation_setup does not exist

        Returns:
            the symmetry measures of a given coordination_geometry for a set of permutations
        """
    csms = []
    permutations = []
    algos = []
    local2perfect_maps = []
    perfect2local_maps = []
    for algo in coordination_geometry.algorithms:
        if algo.algorithm_type == SEPARATION_PLANE:
            cgsm = self.coordination_geometry_symmetry_measures_separation_plane_optim(coordination_geometry, algo, points_perfect=points_perfect, nb_set=nb_set, optimization=optimization)
            csm, perm, algo, local2perfect_map, perfect2local_map = cgsm
            csms.extend(csm)
            permutations.extend(perm)
            algos.extend(algo)
            local2perfect_maps.extend(local2perfect_map)
            perfect2local_maps.extend(perfect2local_map)
    return (csms, permutations, algos, local2perfect_maps, perfect2local_maps)