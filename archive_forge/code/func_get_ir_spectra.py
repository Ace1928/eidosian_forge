from __future__ import annotations
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.core.spectrum import Spectrum
from pymatgen.core.structure import Structure
from pymatgen.util.plotting import add_fig_kwargs
from pymatgen.vis.plotters import SpectrumPlotter
def get_ir_spectra(self, broad: list | float=5e-05, emin: float=0, emax: float | None=None, divs: int=500) -> tuple:
    """The IR spectra is obtained for the different directions.

        Args:
            broad: a list of broadenings or a single broadening for the phonon peaks
            emin (float): minimum energy in which to obtain the spectra. Defaults to 0.
            emax (float): maximum energy in which to obtain the spectra. Defaults to None.
            divs: number of frequency samples between emin and emax

        Returns:
            frequencies: divs array with the frequencies at which the
                         dielectric tensor is calculated
            dielectric_tensor: divsx3x3 numpy array with the dielectric tensor
                         for the range of frequencies
        """
    if isinstance(broad, float):
        broad = [broad] * self.nph_freqs
    if isinstance(broad, list) and len(broad) != self.nph_freqs:
        raise ValueError('The number of elements in the broad_list is not the same as the number of frequencies')
    if emax is None:
        emax = self.max_phfreq + max(broad) * 20
    frequencies = np.linspace(emin, emax, divs)
    na = np.newaxis
    dielectric_tensor = np.zeros((divs, 3, 3), dtype=complex)
    for i in range(3, len(self.ph_freqs_gamma)):
        g = broad[i] * self.ph_freqs_gamma[i]
        num = self.oscillator_strength[i, :, :]
        den = self.ph_freqs_gamma[i] ** 2 - frequencies[:, na, na] ** 2 - 1j * g
        dielectric_tensor += num / den
    dielectric_tensor += self.epsilon_infinity[na, :, :]
    return (frequencies, dielectric_tensor)