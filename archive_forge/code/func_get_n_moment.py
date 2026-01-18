from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def get_n_moment(self, n: int, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None, center: bool=True) -> float:
    """Get the nth moment of the DOS centered around the orbital-projected band center, defined as
            int_{-inf}^{+inf} rho(E)*(E-E_center)^n dE/int_{-inf}^{+inf} rho(E) dE
        where n is the order, E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level. If center is False,
        then the E_center reference is not used.

        Args:
            n: The order for the moment
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.
            center: Take moments with respect to the band center

        Returns:
            Orbital-projected nth moment in eV
        """
    if elements and sites:
        raise ValueError('Both element and site cannot be specified.')
    densities: Mapping[Spin, ArrayLike] = {}
    if elements:
        for idx, el in enumerate(elements):
            spd_dos = self.get_element_spd_dos(el)[band]
            densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
        dos = Dos(self.efermi, self.energies, densities)
    elif sites:
        for idx, site in enumerate(sites):
            spd_dos = self.get_site_spd_dos(site)[band]
            densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
        dos = Dos(self.efermi, self.energies, densities)
    else:
        dos = self.get_spd_dos()[band]
    energies = dos.energies - dos.efermi
    dos_densities = dos.get_densities(spin=spin)
    if erange:
        dos_densities = dos_densities[(energies >= erange[0]) & (energies <= erange[1])]
        energies = energies[(energies >= erange[0]) & (energies <= erange[1])]
    if center:
        band_center = self.get_band_center(elements=elements, sites=sites, band=band, spin=spin, erange=erange)
        p = energies - band_center
    else:
        p = energies
    return np.trapz(p ** n * dos_densities, x=energies) / np.trapz(dos_densities, x=energies)