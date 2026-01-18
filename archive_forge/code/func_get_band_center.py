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
def get_band_center(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
    """Compute the orbital-projected band center, defined as the first moment
        relative to the Fermi level
            int_{-inf}^{+inf} rho(E)*E dE/int_{-inf}^{+inf} rho(E) dE
        based on the work of Hammer and Norskov, Surf. Sci., 343 (1995) where the
        limits of the integration can be modified by erange and E is the set
        of energies taken with respect to the Fermi level. Note that the band center
        is often highly sensitive to the selected erange.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: band center in eV, often denoted epsilon_d for the d-band center
        """
    return self.get_n_moment(1, elements=elements, sites=sites, band=band, spin=spin, erange=erange, center=False)