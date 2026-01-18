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
def get_neighb_voronoi_indices(self, permutation):
    """Get indices in the detailed_voronoi corresponding to the current permutation.

            Args:
                permutation: Current permutation for which the indices in the detailed_voronoi are needed.

            Returns:
                list[int]: indices in the detailed_voronoi.
            """
    return [self.site_voronoi_indices[ii] for ii in permutation]