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
def plot_csm_and_maps(self, isite, max_csm=8.0):
    """
        Plotting of the coordination numbers of a given site for all the distfactor/angfactor parameters. If the
        chemical environments are given, a color map is added to the plot, with the lowest continuous symmetry measure
        as the value for the color of that distfactor/angfactor set.

        Args:
            isite: Index of the site for which the plot has to be done
            max_csm: Maximum continuous symmetry measure to be shown.
        """
    fig = self.get_csm_and_maps(isite=isite, max_csm=max_csm)
    if fig is None:
        return
    plt.show()
    return