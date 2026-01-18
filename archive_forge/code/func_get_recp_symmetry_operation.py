from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def get_recp_symmetry_operation(self, symprec: float=0.01) -> list:
    """Find the symmetric operations of the reciprocal lattice,
        to be used for hkl transformations.

        Args:
            symprec: default is 0.001.
        """
    recp_lattice = self.reciprocal_lattice_crystallographic
    recp_lattice = recp_lattice.scale(1)
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    recp = Structure(recp_lattice, ['H'], [[0, 0, 0]])
    analyzer = SpacegroupAnalyzer(recp, symprec=symprec)
    return analyzer.get_symmetry_operations()