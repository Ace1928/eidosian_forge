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
def symmetry_measure(points_distorted, points_perfect):
    """
    Computes the continuous symmetry measure of the (distorted) set of points "points_distorted" with respect to the
    (perfect) set of points "points_perfect".

    Args:
        points_distorted: List of points describing a given (distorted) polyhedron for which the symmetry measure
            has to be computed with respect to the model polyhedron described by the list of points
            "points_perfect".
        points_perfect: List of "perfect" points describing a given model polyhedron.

    Returns:
        The continuous symmetry measure of the distorted polyhedron with respect to the perfect polyhedron.
    """
    if len(points_distorted) == 1:
        return {'symmetry_measure': 0.0, 'scaling_factor': None, 'rotation_matrix': None}
    rot = find_rotation(points_distorted=points_distorted, points_perfect=points_perfect)
    scaling_factor, rotated_coords, points_perfect = find_scaling_factor(points_distorted=points_distorted, points_perfect=points_perfect, rot=rot)
    rotated_coords = scaling_factor * rotated_coords
    diff = points_perfect - rotated_coords
    num = np.tensordot(diff, diff)
    denom = np.tensordot(points_perfect, points_perfect)
    return {'symmetry_measure': num / denom * 100.0, 'scaling_factor': scaling_factor, 'rotation_matrix': rot}