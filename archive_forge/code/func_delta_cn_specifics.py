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
def delta_cn_specifics(cls, delta_csm_mins=None, delta_csm_maxs=None, function='smootherstep', symmetry_measure_type='csm_wcs_ctwcc', effective_csm_estimator=DEFAULT_EFFECTIVE_CSM_ESTIMATOR):
    """Initialize DeltaCSMNbSetWeight from specific coordination number differences.

        Args:
            delta_csm_mins: Minimums for each coordination number.
            delta_csm_maxs: Maximums for each coordination number.
            function: Ratio function used.
            symmetry_measure_type: Type of symmetry measure to be used.
            effective_csm_estimator: Ratio function used for the effective CSM (comparison between neighbors sets).

        Returns:
            DeltaCSMNbSetWeight.
        """
    if delta_csm_mins is None or delta_csm_maxs is None:
        delta_cn_weight_estimators = {dcn: {'function': function, 'options': {'delta_csm_min': 0.25 + dcn * 0.25, 'delta_csm_max': 5.0 + dcn * 0.25}} for dcn in range(1, 13)}
    else:
        delta_cn_weight_estimators = {dcn: {'function': function, 'options': {'delta_csm_min': delta_csm_mins[dcn - 1], 'delta_csm_max': delta_csm_maxs[dcn - 1]}} for dcn in range(1, 13)}
    return cls(effective_csm_estimator=effective_csm_estimator, weight_estimator={'function': function, 'options': {'delta_csm_min': delta_cn_weight_estimators[12]['options']['delta_csm_min'], 'delta_csm_max': delta_cn_weight_estimators[12]['options']['delta_csm_max']}}, delta_cn_weight_estimators=delta_cn_weight_estimators, symmetry_measure_type=symmetry_measure_type)