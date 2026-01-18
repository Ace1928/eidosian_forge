from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def angle_plateau(self):
    """Returns the angles plateau's for this NeighborsSet."""
    all_nbs_normalized_angles_sorted = sorted((nb['normalized_angle'] for nb in self.voronoi))
    minang = np.min(self.normalized_angles)
    for nb in self.voronoi:
        print(nb)
    plateau = None
    for iang, ang in enumerate(all_nbs_normalized_angles_sorted):
        if np.isclose(ang, minang, rtol=0.0, atol=self.detailed_voronoi.normalized_angle_tolerance):
            plateau = minang if iang == 0 else minang - all_nbs_normalized_angles_sorted[iang - 1]
            break
    if plateau is None:
        raise ValueError('Plateau not found ...')
    return plateau