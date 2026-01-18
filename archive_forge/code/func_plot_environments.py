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
def plot_environments(self, isite, plot_type=None, title='Coordination numbers', max_dist=2.0, figsize=None, strategy=None):
    """
        Plotting of the coordination numbers of a given site for all the distfactor/angfactor parameters. If the
        chemical environments are given, a color map is added to the plot, with the lowest continuous symmetry measure
        as the value for the color of that distfactor/angfactor set.

        Args:
            isite: Index of the site for which the plot has to be done.
            plot_type: How to plot the coordinations.
            title: Title for the figure.
            max_dist: Maximum distance to be plotted when the plotting of the distance is set to 'initial_normalized'
                or 'initial_real' (Warning: this is not the same meaning in both cases! In the first case, the
                closest atom lies at a "normalized" distance of 1.0 so that 2.0 means refers to this normalized
                distance while in the second case, the real distance is used).
            figsize: Size of the figure.
            strategy: Whether to plot information about one of the Chemenv Strategies.
        """
    fig, _ax = self.get_environments_figure(isite=isite, plot_type=plot_type, title=title, max_dist=max_dist, figsize=figsize, strategy=strategy)
    if fig is None:
        return
    fig.show()