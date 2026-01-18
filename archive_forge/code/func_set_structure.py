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
def set_structure(self, lattice: Lattice, species, coords, coords_are_cartesian):
    """
        Sets up the pymatgen structure for which the coordination geometries have to be identified starting from the
        lattice, the species and the coordinates

        Args:
            lattice: The lattice of the structure
            species: The species on the sites
            coords: The coordinates of the sites
            coords_are_cartesian: If set to True, the coordinates are given in Cartesian coordinates.
        """
    self.setup_structure(Structure(lattice, species, coords, coords_are_cartesian))