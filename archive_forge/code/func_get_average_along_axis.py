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
def get_average_along_axis(self, ind):
    """
        Get the averaged total of the volumetric data a certain axis direction.
        For example, useful for visualizing Hartree Potentials from a LOCPOT
        file.

        Args:
            ind (int): Index of axis.

        Returns:
            Average total along axis
        """
    m = self.data['total']
    ng = self.dim
    if ind == 0:
        total = np.sum(np.sum(m, axis=1), 1)
    elif ind == 1:
        total = np.sum(np.sum(m, axis=0), 1)
    else:
        total = np.sum(np.sum(m, axis=0), 0)
    return total / ng[(ind + 1) % 3] / ng[(ind + 2) % 3]