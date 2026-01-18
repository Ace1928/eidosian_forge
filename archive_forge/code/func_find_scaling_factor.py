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
def find_scaling_factor(points_distorted, points_perfect, rot):
    """
    This finds the scaling factor between the (distorted) set of points "points_distorted" and the
    (perfect) set of points "points_perfect" in a least-square sense.

    Args:
        points_distorted: List of points describing a given (distorted) polyhedron for
            which the scaling factor has to be obtained.
        points_perfect: List of "perfect" points describing a given model polyhedron.
        rot: The rotation matrix

    Returns:
        The scaling factor between the two structures and the rotated set of (distorted) points.
    """
    rotated_coords = np.matmul(rot, points_distorted.T).T
    num = np.tensordot(rotated_coords, points_perfect)
    denom = np.tensordot(rotated_coords, rotated_coords)
    return (num / denom, rotated_coords, points_perfect)