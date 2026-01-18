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
def linear_add(self, other, scale_factor=1.0):
    """
        Method to do a linear sum of volumetric objects. Used by + and -
        operators as well. Returns a VolumetricData object containing the
        linear sum.

        Args:
            other (VolumetricData): Another VolumetricData object
            scale_factor (float): Factor to scale the other data by.

        Returns:
            VolumetricData corresponding to self + scale_factor * other.
        """
    if self.structure != other.structure:
        warnings.warn('Structures are different. Make sure you know what you are doing...')
    if list(self.data) != list(other.data):
        raise ValueError('Data have different keys! Maybe one is spin-polarized and the other is not?')
    data = {}
    for k in self.data:
        data[k] = self.data[k] + scale_factor * other.data[k]
    new = deepcopy(self)
    new.data = data
    new.data_aug = {}
    return new