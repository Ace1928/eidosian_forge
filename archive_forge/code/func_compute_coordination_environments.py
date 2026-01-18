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
def compute_coordination_environments(self, structure, indices=None, only_cations=True, strategy=DEFAULT_STRATEGY, valences='bond-valence-analysis', initial_structure_environments=None):
    """
        Args:
            structure:
            indices:
            only_cations:
            strategy:
            valences:
            initial_structure_environments:
        """
    self.setup_structure(structure=structure)
    if valences == 'bond-valence-analysis':
        bva = BVAnalyzer()
        try:
            vals = bva.get_valences(structure=structure)
        except ValueError:
            vals = 'undefined'
    elif valences == 'undefined':
        vals = valences
    else:
        len_vals, len_sites = (len(valences), len(structure))
        if len_vals != len_sites:
            raise ValueError(f'Valences ({len_vals}) do not match the number of sites in the structure ({len_sites})')
        vals = valences
    se = self.compute_structure_environments(only_cations=only_cations, only_indices=indices, valences=vals, initial_structure_environments=initial_structure_environments)
    lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
    return lse.coordination_environments