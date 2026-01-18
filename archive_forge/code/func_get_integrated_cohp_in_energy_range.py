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
def get_integrated_cohp_in_energy_range(cohp, label, orbital=None, energy_range=None, relative_E_Fermi=True, summed_spin_channels=True):
    """Method that can integrate completecohp objects which include data on integrated COHPs
    Args:
        cohp: CompleteCOHP object
        label: label of the COHP data
        orbital: If not None, a orbital resolved integrated COHP will be returned
        energy_range:   if None, returns icohp value at Fermi level;
                        if float, integrates from this float up to the Fermi level;
                        if [float,float], will integrate in between
        relative_E_Fermi: if True, energy scale with E_Fermi at 0 eV is chosen
        summed_spin_channels: if True, Spin channels will be summed.

    Returns:
        float indicating the integrated COHP if summed_spin_channels==True, otherwise dict of the following form {
        Spin.up:float, Spin.down:float}
    """
    summedicohp = {}
    if orbital is None:
        icohps = cohp.all_cohps[label].get_icohp(spin=None)
        if summed_spin_channels and Spin.down in icohps:
            summedicohp[Spin.up] = icohps[Spin.up] + icohps[Spin.down]
        else:
            summedicohp = icohps
    else:
        icohps = cohp.get_orbital_resolved_cohp(label=label, orbitals=orbital).icohp
        if summed_spin_channels and Spin.down in icohps:
            summedicohp[Spin.up] = icohps[Spin.up] + icohps[Spin.down]
        else:
            summedicohp = icohps
    if energy_range is None:
        energies_corrected = cohp.energies - cohp.efermi
        spl_spinup = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.up], ext=0)
        if not summed_spin_channels and Spin.down in icohps:
            spl_spindown = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.down], ext=0)
            return {Spin.up: spl_spinup(0.0), Spin.down: spl_spindown(0.0)}
        if summed_spin_channels:
            return spl_spinup(0.0)
        return {Spin.up: spl_spinup(0.0)}
    if isinstance(energy_range, float):
        if relative_E_Fermi:
            energies_corrected = cohp.energies - cohp.efermi
            spl_spinup = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.up], ext=0)
            if not summed_spin_channels and Spin.down in icohps:
                spl_spindown = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.down], ext=0)
                return {Spin.up: spl_spinup(0) - spl_spinup(energy_range), Spin.down: spl_spindown(0) - spl_spindown(energy_range)}
            if summed_spin_channels:
                return spl_spinup(0) - spl_spinup(energy_range)
            return {Spin.up: spl_spinup(0) - spl_spinup(energy_range)}
        energies_corrected = cohp.energies
        spl_spinup = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.up], ext=0)
        if not summed_spin_channels and Spin.down in icohps:
            spl_spindown = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.down], ext=0)
            return {Spin.up: spl_spinup(cohp.efermi) - spl_spinup(energy_range), Spin.down: spl_spindown(cohp.efermi) - spl_spindown(energy_range)}
        if summed_spin_channels:
            return spl_spinup(cohp.efermi) - spl_spinup(energy_range)
        return {Spin.up: spl_spinup(cohp.efermi) - spl_spinup(energy_range)}
    energies_corrected = cohp.energies - cohp.efermi if relative_E_Fermi else cohp.energies
    spl_spinup = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.up], ext=0)
    if not summed_spin_channels and Spin.down in icohps:
        spl_spindown = InterpolatedUnivariateSpline(energies_corrected, summedicohp[Spin.down], ext=0)
        return {Spin.up: spl_spinup(energy_range[1]) - spl_spinup(energy_range[0]), Spin.down: spl_spindown(energy_range[1]) - spl_spindown(energy_range[0])}
    if summed_spin_channels:
        return spl_spinup(energy_range[1]) - spl_spinup(energy_range[0])
    return {Spin.up: spl_spinup(energy_range[1]) - spl_spinup(energy_range[0])}