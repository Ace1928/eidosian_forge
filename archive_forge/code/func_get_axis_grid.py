from __future__ import annotations
import itertools
import json
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from scipy.interpolate import RegularGridInterpolator
from pymatgen.core import Element, Site, Structure
from pymatgen.core.units import ang_to_bohr, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
def get_axis_grid(self, ind):
    """
        Returns the grid for a particular axis.

        Args:
            ind (int): Axis index.
        """
    ng = self.dim
    num_pts = ng[ind]
    lengths = self.structure.lattice.abc
    return [i / num_pts * lengths[ind] for i in range(num_pts)]