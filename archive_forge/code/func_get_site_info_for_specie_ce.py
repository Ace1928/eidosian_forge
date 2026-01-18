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
def get_site_info_for_specie_ce(self, specie, ce_symbol):
    """
        Get list of indices that have the given specie with a given Coordination environment.

        Args:
            specie: Species to get.
            ce_symbol: Symbol of the coordination environment to get.

        Returns:
            dict: Keys are 'isites', 'fractions', 'csms' which contain list of indices in the structure
                that have the given specie in the given environment, their fraction and continuous
                symmetry measures.
        """
    element = specie.symbol
    oxi_state = specie.oxi_state
    isites = []
    csms = []
    fractions = []
    for isite, site in enumerate(self.structure):
        if element in [sp.symbol for sp in site.species] and (self.valences == 'undefined' or oxi_state == self.valences[isite]):
            for ce_dict in self.coordination_environments[isite]:
                if ce_symbol == ce_dict['ce_symbol']:
                    isites.append(isite)
                    csms.append(ce_dict['csm'])
                    fractions.append(ce_dict['ce_fraction'])
    return {'isites': isites, 'fractions': fractions, 'csms': csms}