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
def get_doping(self, fermi_level: float, temperature: float) -> float:
    """Calculate the doping (majority carrier concentration) at a given
        Fermi level  and temperature. A simple Left Riemann sum is used for
        integrating the density of states over energy & equilibrium Fermi-Dirac
        distribution.

        Args:
            fermi_level: The fermi_level level in eV.
            temperature: The temperature in Kelvin.

        Returns:
            The doping concentration in units of 1/cm^3. Negative values
            indicate that the majority carriers are electrons (n-type doping)
            whereas positive values indicates the majority carriers are holes
            (p-type doping).
        """
    cb_integral = np.sum(self.tdos[self.idx_cbm:] * f0(self.energies[self.idx_cbm:], fermi_level, temperature) * self.de[self.idx_cbm:], axis=0)
    vb_integral = np.sum(self.tdos[:self.idx_vbm + 1] * f0(-self.energies[:self.idx_vbm + 1], -fermi_level, temperature) * self.de[:self.idx_vbm + 1], axis=0)
    return (vb_integral - cb_integral) / (self.volume * self.A_to_cm ** 3)