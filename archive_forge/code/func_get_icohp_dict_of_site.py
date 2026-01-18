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
def get_icohp_dict_of_site(self, site, minsummedicohp=None, maxsummedicohp=None, minbondlength=0.0, maxbondlength=8.0, only_bonds_to=None):
    """Get a dict of IcohpValue for a certain site (indicated by integer).

        Args:
            site: integer describing the site of interest, order as in Icohplist.lobster/Icooplist.lobster, starts at 0
            minsummedicohp: float, minimal icohp/icoop of the bonds that are considered. It is the summed ICOHP value
                from both spin channels for spin polarized cases
            maxsummedicohp: float, maximal icohp/icoop of the bonds that are considered. It is the summed ICOHP value
                from both spin channels for spin polarized cases
            minbondlength: float, defines the minimum of the bond lengths of the bonds
            maxbondlength: float, defines the maximum of the bond lengths of the bonds
            only_bonds_to: list of strings describing the bonding partners that are allowed, e.g. ['O']

        Returns:
            dict of IcohpValues, the keys correspond to the values from the initial list_labels
        """
    new_icohp_dict = {}
    for key, value in self._icohplist.items():
        atomnumber1 = int(re.split('(\\d+)', value._atom1)[1]) - 1
        atomnumber2 = int(re.split('(\\d+)', value._atom2)[1]) - 1
        if site in (atomnumber1, atomnumber2):
            if site == atomnumber2:
                save = value._atom1
                value._atom1 = value._atom2
                value._atom2 = save
            second_test = True if only_bonds_to is None else re.split('(\\d+)', value._atom2)[0] in only_bonds_to
            if value._length >= minbondlength and value._length <= maxbondlength and second_test:
                if minsummedicohp is not None:
                    if value.summed_icohp >= minsummedicohp:
                        if maxsummedicohp is not None:
                            if value.summed_icohp <= maxsummedicohp:
                                new_icohp_dict[key] = value
                        else:
                            new_icohp_dict[key] = value
                elif maxsummedicohp is not None:
                    if value.summed_icohp <= maxsummedicohp:
                        new_icohp_dict[key] = value
                else:
                    new_icohp_dict[key] = value
    return new_icohp_dict