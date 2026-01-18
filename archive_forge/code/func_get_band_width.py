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
def get_band_width(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
    """Get the orbital-projected band width, defined as the square root of the second moment
            sqrt(int_{-inf}^{+inf} rho(E)*(E-E_center)^2 dE/int_{-inf}^{+inf} rho(E) dE)
        where E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level.
        Note that the band width is often highly sensitive to the selected erange.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: Orbital-projected band width in eV
        """
    return np.sqrt(self.get_n_moment(2, elements=elements, sites=sites, band=band, spin=spin, erange=erange))