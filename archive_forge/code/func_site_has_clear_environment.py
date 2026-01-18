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
def site_has_clear_environment(self, isite, conditions=None):
    """
        Whether a given site has a "clear" environments.

        A "clear" environment is somewhat arbitrary. You can pass (multiple) conditions, e.g. the environment should
        have a continuous symmetry measure lower than this, a fraction higher than that, ...

        Args:
            isite: Index of the site.
            conditions: Conditions to be checked for an environment to be "clear".

        Returns:
            bool: True if the site has a clear environment.
        """
    if self.coordination_environments[isite] is None:
        raise ValueError(f'Coordination environments have not been determined for site {isite}')
    if conditions is None:
        return len(self.coordination_environments[isite]) == 1
    ce = max(self.coordination_environments[isite], key=lambda x: x['ce_fraction'])
    for condition in conditions:
        target = condition['target']
        if target == 'ce_fraction':
            if ce[target] < condition['minvalue']:
                return False
        elif target == 'csm':
            if ce[target] > condition['maxvalue']:
                return False
        elif target == 'number_of_ces':
            if ce[target] > condition['maxnumber']:
                return False
        else:
            raise ValueError(f'Target {target!r} for condition of clear environment is not allowed')
    return True