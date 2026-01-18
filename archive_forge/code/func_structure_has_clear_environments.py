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
def structure_has_clear_environments(self, conditions=None, skip_none=True, skip_empty=False):
    """
        Whether all sites in a structure have "clear" environments.

        Args:
            conditions: Conditions to be checked for an environment to be "clear".
            skip_none: Whether to skip sites for which no environments have been computed.
            skip_empty: Whether to skip sites for which no environments could be found.

        Returns:
            bool: True if all the sites in the structure have clear environments.
        """
    for isite in range(len(self.structure)):
        if self.coordination_environments[isite] is None:
            if skip_none:
                continue
            return False
        if len(self.coordination_environments[isite]) == 0:
            if skip_empty:
                continue
            return False
        if not self.site_has_clear_environment(isite=isite, conditions=conditions):
            return False
    return True