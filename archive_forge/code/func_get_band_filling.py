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
def get_band_filling(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None) -> float:
    """Compute the orbital-projected band filling, defined as the zeroth moment
        up to the Fermi level.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.

        Returns:
            float: band filling in eV, often denoted f_d for the d-band
        """
    if elements and sites:
        raise ValueError('Both element and site cannot be specified.')
    densities: dict[Spin, ArrayLike] = {}
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
    energies = dos.energies - dos.efermi
    return np.trapz(dos_densities[energies < 0], x=energies[energies < 0]) / np.trapz(dos_densities, x=energies)