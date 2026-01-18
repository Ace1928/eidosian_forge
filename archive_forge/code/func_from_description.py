from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@classmethod
def from_description(cls, dct: dict) -> Self:
    """Initialize weights from description.

        Args:
            dct (dict): Dictionary description.

        Returns:
            CNBiasNbSetWeight.
        """
    if dct['type'] == 'linearly_equidistant':
        return cls.linearly_equidistant(weight_cn1=dct['weight_cn1'], weight_cn13=dct['weight_cn13'])
    if dct['type'] == 'geometrically_equidistant':
        return cls.geometrically_equidistant(weight_cn1=dct['weight_cn1'], weight_cn13=dct['weight_cn13'])
    if dct['type'] == 'explicit':
        return cls.explicit(cn_weights=dct['cn_weights'])
    raise RuntimeError('Cannot initialize Weights.')