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
def add_neighbors_set(self, isite, nb_set):
    """
        Adds a neighbor set to the list of neighbors sets for this site.

        Args:
            isite: Index of the site under consideration.
            nb_set: NeighborsSet to be added.
        """
    if self.neighbors_sets[isite] is None:
        self.neighbors_sets[isite] = {}
        self.ce_list[isite] = {}
    cn = len(nb_set)
    if cn not in self.neighbors_sets[isite]:
        self.neighbors_sets[isite][cn] = []
        self.ce_list[isite][cn] = []
    try:
        nb_set_index = self.neighbors_sets[isite][cn].index(nb_set)
        self.neighbors_sets[isite][cn][nb_set_index].add_source(nb_set.source)
    except ValueError:
        self.neighbors_sets[isite][cn].append(nb_set)
        self.ce_list[isite][cn].append(None)