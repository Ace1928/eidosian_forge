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
@classmethod
def from_cg(cls, cg, centering_type='standard', include_central_site_in_centroid=False) -> Self:
    """
        Args:
            cg:
            centering_type:
            include_central_site_in_centroid:
        """
    central_site = cg.get_central_site()
    bare_coords = [np.array(pt, float) for pt in cg.points]
    return cls(central_site=central_site, bare_coords=bare_coords, centering_type=centering_type, include_central_site_in_centroid=include_central_site_in_centroid)