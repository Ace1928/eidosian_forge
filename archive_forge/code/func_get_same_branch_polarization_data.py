from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """
        Get same branch dipole moment (convert_to_muC_per_cm2=False)
        or polarization for given polarization data (convert_to_muC_per_cm2=True).

        Polarization is a lattice vector, meaning it is only defined modulo the
        quantum of polarization:

            P = P_0 + \\\\sum_i \\\\frac{n_i e R_i}{\\\\Omega}

        where n_i is an integer, e is the charge of the electron in microCoulombs,
        R_i is a lattice vector, and \\\\Omega is the unit cell volume in cm**3
        (giving polarization units of microCoulomb per centimeter**2).

        The quantum of the dipole moment in electron Angstroms (as given by VASP) is:

            \\\\sum_i n_i e R_i

        where e, the electron charge, is 1 and R_i is a lattice vector, and n_i is an integer.

        Given N polarization calculations in order from nonpolar to polar, this algorithm
        minimizes the distance between adjacent polarization images. To do this, it
        constructs a polarization lattice for each polarization calculation using the
        pymatgen.core.structure class and calls the get_nearest_site method to find the
        image of a given polarization lattice vector that is closest to the previous polarization
        lattice vector image.

        Note, using convert_to_muC_per_cm2=True and all_in_polar=True calculates the "proper
        polarization" (meaning the change in polarization does not depend on the choice of
        polarization branch) while convert_to_muC_per_cm2=True and all_in_polar=False calculates
        the "improper polarization" (meaning the change in polarization does depend on the choice
        of branch). As one might guess from the names. We recommend calculating the "proper
        polarization".

        convert_to_muC_per_cm2: convert polarization from electron * Angstroms to
            microCoulomb per centimeter**2
        all_in_polar: convert polarization to be in polar (final structure) polarization lattice
        """
    p_elec, p_ion = self.get_pelecs_and_pions()
    p_tot = p_elec + p_ion
    p_tot = np.array(p_tot)
    lattices = [s.lattice for s in self.structures]
    volumes = np.array([s.lattice.volume for s in self.structures])
    n_elecs = len(p_elec)
    e_to_muC = -1.6021766e-13
    cm2_to_A2 = 1e+16
    units = 1 / np.array(volumes)
    units *= e_to_muC * cm2_to_A2
    if convert_to_muC_per_cm2 and (not all_in_polar):
        p_tot = np.multiply(units.T[:, np.newaxis], p_tot)
        for idx in range(n_elecs):
            lattice = lattices[idx]
            lattices[idx] = Lattice.from_parameters(*np.array(lattice.lengths) * units.ravel()[idx], *lattice.angles)
    elif convert_to_muC_per_cm2 and all_in_polar:
        abc = [lattice.abc for lattice in lattices]
        abc = np.array(abc)
        p_tot /= abc
        p_tot *= abc[-1] / volumes[-1] * e_to_muC * cm2_to_A2
        for idx in range(n_elecs):
            lattice = lattices[-1]
            lattices[idx] = Lattice.from_parameters(*np.array(lattice.lengths) * units.ravel()[-1], *lattice.angles)
    d_structs = []
    sites = []
    for idx in range(n_elecs):
        lattice = lattices[idx]
        frac_coord = np.divide(np.array([p_tot[idx]]), np.array(lattice.lengths))
        struct = Structure(lattice, ['C'], [np.array(frac_coord).ravel()])
        d_structs.append(struct)
        site = struct[0]
        prev_site = [0, 0, 0] if idx == 0 else sites[-1].coords
        new_site = get_nearest_site(struct, prev_site, site)
        sites.append(new_site[0])
    adjust_pol = []
    for site, struct in zip(sites, d_structs):
        adjust_pol.append(np.multiply(site.frac_coords, np.array(struct.lattice.lengths)).ravel())
    return np.array(adjust_pol)