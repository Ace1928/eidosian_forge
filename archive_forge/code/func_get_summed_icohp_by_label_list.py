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
def get_summed_icohp_by_label_list(self, label_list, divisor=1.0, summed_spin_channels=True, spin=Spin.up):
    """Get the sum of several ICOHP values that are indicated by a list of labels
        (labels of the bonds are the same as in ICOHPLIST/ICOOPLIST).

        Args:
            label_list: list of labels of the ICOHPs/ICOOPs that should be summed
            divisor: is used to divide the sum
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed
            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned

        Returns:
            float that is a sum of all ICOHPs/ICOOPs as indicated with label_list
        """
    sum_icohp = 0
    for label in label_list:
        icohp_here = self._icohplist[label]
        if icohp_here.num_bonds != 1:
            warnings.warn('One of the ICOHP values is an average over bonds. This is currently not considered.')
        if icohp_here._is_spin_polarized:
            if summed_spin_channels:
                sum_icohp = sum_icohp + icohp_here.summed_icohp
            else:
                sum_icohp = sum_icohp + icohp_here.icohpvalue(spin)
        else:
            sum_icohp = sum_icohp + icohp_here.icohpvalue(spin)
    return sum_icohp / divisor