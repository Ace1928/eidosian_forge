from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def is_polar(self, tol_dipole_per_unit_area: float=0.001) -> bool:
    """Check if the Slab is polar by computing the normalized dipole per unit area.
        Normalized dipole per unit area is used as it is more reliable than
        using the absolute value, which varies with surface area.

        Note that the Slab must be oxidation state decorated for this to work properly.
        Otherwise, the Slab will always have a dipole moment of 0.

        Args:
            tol_dipole_per_unit_area (float): A tolerance above which the Slab is
                considered polar.
        """
    dip_per_unit_area = self.dipole / self.surface_area
    return np.linalg.norm(dip_per_unit_area) > tol_dipole_per_unit_area