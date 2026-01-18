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
def stats_article_weights_parameters(cls):
    """Initialize strategy used in the statistics article."""
    self_csm_weight = SelfCSMNbSetWeight(weight_estimator={'function': 'power2_decreasing_exp', 'options': {'max_csm': 8.0, 'alpha': 1}})
    surface_definition = {'type': 'standard_elliptic', 'distance_bounds': {'lower': 1.15, 'upper': 2.0}, 'angle_bounds': {'lower': 0.05, 'upper': 0.75}}
    da_area_weight = DistanceAngleAreaNbSetWeight(weight_type='has_intersection', surface_definition=surface_definition, nb_sets_from_hints='fallback_to_source', other_nb_sets='0_weight', additional_condition=DistanceAngleAreaNbSetWeight.AC.ONLY_ACB)
    symmetry_measure_type = 'csm_wcs_ctwcc'
    delta_weight = DeltaCSMNbSetWeight.delta_cn_specifics()
    bias_weight = angle_weight = nad_weight = None
    return cls(dist_ang_area_weight=da_area_weight, self_csm_weight=self_csm_weight, delta_csm_weight=delta_weight, cn_bias_weight=bias_weight, angle_weight=angle_weight, normalized_angle_distance_weight=nad_weight, symmetry_measure_type=symmetry_measure_type)