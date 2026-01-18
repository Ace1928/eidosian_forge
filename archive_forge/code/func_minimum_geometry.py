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
def minimum_geometry(self, symmetry_measure_type=None, max_csm=None):
    """
        Returns the geometry with the minimum continuous symmetry measure of this ChemicalEnvironments.

        Returns:
            tuple (symbol, csm) with symbol being the geometry with the minimum continuous symmetry measure and
            csm being the continuous symmetry measure associated to it.

        Raises:
            ValueError if no coordination geometry is found in this ChemicalEnvironments object.
        """
    if len(self.coord_geoms) == 0:
        return None
    cglist = list(self.coord_geoms)
    if symmetry_measure_type is None:
        csms = np.array([self.coord_geoms[cg]['other_symmetry_measures']['csm_wcs_ctwcc'] for cg in cglist])
    else:
        csms = np.array([self.coord_geoms[cg]['other_symmetry_measures'][symmetry_measure_type] for cg in cglist])
    csmlist = [self.coord_geoms[cg] for cg in cglist]
    imin = np.argmin(csms)
    if max_csm is not None and csmlist[imin] > max_csm:
        return None
    return (cglist[imin], csmlist[imin])