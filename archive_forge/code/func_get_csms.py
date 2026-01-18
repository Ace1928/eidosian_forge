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
def get_csms(self, isite, mp_symbol) -> list:
    """
        Returns the continuous symmetry measure(s) of site with index isite with respect to the
        perfect coordination environment with mp_symbol. For some environments, a given mp_symbol might not
        be available (if there is no voronoi parameters leading to a number of neighbors corresponding to
        the coordination number of environment mp_symbol). For some environments, a given mp_symbol might
        lead to more than one csm (when two or more different voronoi parameters lead to different neighbors
        but with same number of neighbors).

        Args:
            isite: Index of the site.
            mp_symbol: MP symbol of the perfect environment for which the csm has to be given.

        Returns:
            list[CSM]: for site isite with respect to geometry mp_symbol
        """
    cn = symbol_cn_mapping[mp_symbol]
    if cn not in self.ce_list[isite]:
        return []
    return [envs[mp_symbol] for envs in self.ce_list[isite][cn]]