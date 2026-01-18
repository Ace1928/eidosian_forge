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
def find_rotation(points_distorted, points_perfect):
    """
    This finds the rotation matrix that aligns the (distorted) set of points "points_distorted" with respect to the
    (perfect) set of points "points_perfect" in a least-square sense.

    Args:
        points_distorted: List of points describing a given (distorted) polyhedron for which the rotation that
            aligns these points in a least-square sense to the set of perfect points "points_perfect"
        points_perfect: List of "perfect" points describing a given model polyhedron.

    Returns:
        The rotation matrix.
    """
    H = np.matmul(points_distorted.T, points_perfect)
    U, _S, Vt = svd(H)
    return np.matmul(Vt.T, U.T)