from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def setup_parameters(self, centering_type='standard', include_central_site_in_centroid=False, bva_distance_scale_factor=None, structure_refinement=STRUCTURE_REFINEMENT_REFINED, spg_analyzer_options=None):
    """
        Setup of the parameters for the coordination geometry finder. A reference point for the geometries has to be
        chosen. This can be the centroid of the structure (including or excluding the atom for which the coordination
        geometry is looked for) or the atom itself. In the 'standard' centering_type, the reference point is the central
        atom for coordination numbers 1, 2, 3 and 4 and the centroid for coordination numbers > 4.

        Args:
            centering_type: Type of the reference point (centering) 'standard', 'centroid' or 'central_site'
            include_central_site_in_centroid: In case centering_type is 'centroid', the central site is included if
                this value is set to True.
            bva_distance_scale_factor: Scaling factor for the bond valence analyzer (this might be different whether
                the structure is an experimental one, an LDA or a GGA relaxed one, or any other relaxation scheme (where
                under- or over-estimation of bond lengths is known).
            structure_refinement: Refinement of the structure. Can be "none", "refined" or "symmetrized".
            spg_analyzer_options: Options for the SpaceGroupAnalyzer (dictionary specifying "symprec"
                and "angle_tolerance". See pymatgen's SpaceGroupAnalyzer for more information.
        """
    self.centering_type = centering_type
    self.include_central_site_in_centroid = include_central_site_in_centroid
    if bva_distance_scale_factor is not None:
        self.bva_distance_scale_factor = bva_distance_scale_factor
    else:
        self.bva_distance_scale_factor = self.DEFAULT_BVA_DISTANCE_SCALE_FACTOR
    self.structure_refinement = structure_refinement
    if spg_analyzer_options is None:
        self.spg_analyzer_options = self.DEFAULT_SPG_ANALYZER_OPTIONS
    else:
        self.spg_analyzer_options = spg_analyzer_options