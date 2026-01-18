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
def get_icohp_by_label(self, label, summed_spin_channels=True, spin=Spin.up, orbitals=None):
    """Get an icohp value for a certain bond as indicated by the label (bond labels starting by "1" as in
        ICOHPLIST/ICOOPLIST).

        Args:
            label: label in str format (usually the bond number in Icohplist.lobster/Icooplist.lobster
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed
            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned
            orbitals: List of Orbital or "str(Orbital1)-str(Orbital2)"

        Returns:
            float describing ICOHP/ICOOP value
        """
    icohp_here = self._icohplist[label]
    if orbitals is None:
        if summed_spin_channels:
            return icohp_here.summed_icohp
        return icohp_here.icohpvalue(spin)
    if isinstance(orbitals, list):
        orbitals = f'{orbitals[0]}-{orbitals[1]}'
    if summed_spin_channels:
        return icohp_here.summed_orbital_icohp[orbitals]
    return icohp_here.icohpvalue_orbital(spin=spin, orbitals=orbitals)