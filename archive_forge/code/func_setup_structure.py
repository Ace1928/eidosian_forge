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
def setup_structure(self, structure: Structure):
    """
        Sets up the structure for which the coordination geometries have to be identified. The structure is analyzed
        with the space group analyzer and a refined structure is used

        Args:
            structure: A pymatgen Structure.
        """
    self.initial_structure = structure.copy()
    if self.structure_refinement == self.STRUCTURE_REFINEMENT_NONE:
        self.structure = structure.copy()
        self.spg_analyzer = self.symmetrized_structure = None
    else:
        self.spg_analyzer = SpacegroupAnalyzer(self.initial_structure, symprec=self.spg_analyzer_options['symprec'], angle_tolerance=self.spg_analyzer_options['angle_tolerance'])
        if self.structure_refinement == self.STRUCTURE_REFINEMENT_REFINED:
            self.structure = self.spg_analyzer.get_refined_structure()
            self.symmetrized_structure = None
        elif self.structure_refinement == self.STRUCTURE_REFINEMENT_SYMMETRIZED:
            self.structure = self.spg_analyzer.get_refined_structure()
            self.spg_analyzer_refined = SpacegroupAnalyzer(self.structure, symprec=self.spg_analyzer_options['symprec'], angle_tolerance=self.spg_analyzer_options['angle_tolerance'])
            self.symmetrized_structure = self.spg_analyzer_refined.get_symmetrized_structure()