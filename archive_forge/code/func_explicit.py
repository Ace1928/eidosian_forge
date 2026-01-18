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
def explicit(cls, cn_weights):
    """Initialize weights explicitly for each coordination.

        Args:
            cn_weights: Weights for each coordination.

        Returns:
            CNBiasNbSetWeight.
        """
    initialization_options = {'type': 'explicit'}
    if set(cn_weights) != set(range(1, 14)):
        raise ValueError('Weights should be provided for CN 1 to 13')
    return cls(cn_weights=cn_weights, initialization_options=initialization_options)