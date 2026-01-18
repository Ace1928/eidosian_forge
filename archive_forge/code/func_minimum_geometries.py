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
def minimum_geometries(self, n=None, symmetry_measure_type=None, max_csm=None):
    """
        Returns a list of geometries with increasing continuous symmetry measure in this ChemicalEnvironments object.

        Args:
            n: Number of geometries to be included in the list.

        Returns:
            List of geometries with increasing continuous symmetry measure in this ChemicalEnvironments object.

        Raises:
            ValueError if no coordination geometry is found in this ChemicalEnvironments object.
        """
    cglist = list(self.coord_geoms)
    if symmetry_measure_type is None:
        csms = np.array([self.coord_geoms[cg]['other_symmetry_measures']['csm_wcs_ctwcc'] for cg in cglist])
    else:
        csms = np.array([self.coord_geoms[cg]['other_symmetry_measures'][symmetry_measure_type] for cg in cglist])
    csmlist = [self.coord_geoms[cg] for cg in cglist]
    isorted = np.argsort(csms)
    if max_csm is not None:
        if n is None:
            return [(cglist[ii], csmlist[ii]) for ii in isorted if csms[ii] <= max_csm]
        return [(cglist[ii], csmlist[ii]) for ii in isorted[:n] if csms[ii] <= max_csm]
    if n is None:
        return [(cglist[ii], csmlist[ii]) for ii in isorted]
    return [(cglist[ii], csmlist[ii]) for ii in isorted[:n]]