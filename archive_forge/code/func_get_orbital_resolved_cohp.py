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
def get_orbital_resolved_cohp(self, label, orbitals, summed_spin_channels=False):
    """Get orbital-resolved COHP.

        Args:
            label: bond label (Lobster: labels as in ICOHPLIST/ICOOPLIST.lobster).

            orbitals: The orbitals as a label, or list or tuple of the form
                [(n1, orbital1), (n2, orbital2)]. Orbitals can either be str,
                int, or Orbital.

            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            A Cohp object if CompleteCohp contains orbital-resolved cohp,
            or None if it doesn't.

        Note: It currently assumes that orbitals are str if they aren't the
            other valid types. This is not ideal, but the easiest way to
            avoid unicode issues between python 2 and python 3.
        """
    if self.orb_res_cohp is None:
        return None
    if isinstance(orbitals, (list, tuple)):
        cohp_orbs = [d['orbitals'] for d in self.orb_res_cohp[label].values()]
        orbs = []
        for orbital in orbitals:
            if isinstance(orbital[1], int):
                orbs.append((orbital[0], Orbital(orbital[1])))
            elif isinstance(orbital[1], Orbital):
                orbs.append((orbital[0], orbital[1]))
            elif isinstance(orbital[1], str):
                orbs.append((orbital[0], Orbital[orbital[1]]))
            else:
                raise TypeError('Orbital must be str, int, or Orbital.')
        orb_index = cohp_orbs.index(orbs)
        orb_label = list(self.orb_res_cohp[label])[orb_index]
    elif isinstance(orbitals, str):
        orb_label = orbitals
    else:
        raise TypeError('Orbitals must be str, list, or tuple.')
    try:
        icohp = self.orb_res_cohp[label][orb_label]['ICOHP']
    except KeyError:
        icohp = None
    start_cohp = self.orb_res_cohp[label][orb_label]['COHP']
    start_icohp = icohp
    if summed_spin_channels and Spin.down in start_cohp:
        final_cohp = {}
        final_icohp = {}
        final_cohp[Spin.up] = np.sum([start_cohp[Spin.up], start_cohp[Spin.down]], axis=0)
        if start_icohp is not None:
            final_icohp[Spin.up] = np.sum([start_icohp[Spin.up], start_icohp[Spin.down]], axis=0)
    else:
        final_cohp = start_cohp
        final_icohp = start_icohp
    return Cohp(self.efermi, self.energies, final_cohp, icohp=final_icohp, are_coops=self.are_coops, are_cobis=self.are_cobis)