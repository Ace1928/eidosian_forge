from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lmto import LMTOCopl
from pymatgen.io.lobster import Cohpcar
from pymatgen.util.coord import get_linear_interpolated_value
from pymatgen.util.due import Doi, due
from pymatgen.util.num import round_to_sigfigs
def has_antibnd_states_below_efermi(self, spin=None, limit=0.01):
    """Returns dict indicating if there are antibonding states below the Fermi level depending on the spin
        spin: Spin
        limit: -COHP smaller -limit will be considered.
        """
    populations = self.cohp
    n_energies_below_efermi = len([x for x in self.energies if x <= self.efermi])
    if populations is None:
        return None
    if spin is None:
        dict_to_return = {}
        for sp, cohp_vals in populations.items():
            if max(cohp_vals[0:n_energies_below_efermi]) > limit:
                dict_to_return[sp] = True
            else:
                dict_to_return[sp] = False
    else:
        dict_to_return = {}
        if isinstance(spin, int):
            spin = Spin(spin)
        elif isinstance(spin, str):
            spin = Spin({'up': 1, 'down': -1}[spin.lower()])
        if max(populations[spin][0:n_energies_below_efermi]) > limit:
            dict_to_return[spin] = True
        else:
            dict_to_return[spin] = False
    return dict_to_return