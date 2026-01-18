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
@property
def spin_data(self):
    """
        The data decomposed into actual spin data as {spin: data}.
        Essentially, this provides the actual Spin.up and Spin.down data
        instead of the total and diff. Note that by definition, a
        non-spin-polarized run would have Spin.up data == Spin.down data.
        """
    if not self._spin_data:
        spin_data = {}
        spin_data[Spin.up] = 0.5 * (self.data['total'] + self.data.get('diff', 0))
        spin_data[Spin.down] = 0.5 * (self.data['total'] - self.data.get('diff', 0))
        self._spin_data = spin_data
    return self._spin_data