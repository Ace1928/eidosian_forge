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
def get_dos_fp(self, type: str='summed_pdos', binning: bool=True, min_e: float | None=None, max_e: float | None=None, n_bins: int=256, normalize: bool=True) -> NamedTuple:
    """Generates the DOS fingerprint.

        Based on work of:

        F. Knoop, T. A. r Purcell, M. Scheffler, C. Carbogno, J. Open Source Softw. 2020, 5, 2671.
        Source - https://gitlab.com/vibes-developers/vibes/-/tree/master/vibes/materials_fp
        Copyright (c) 2020 Florian Knoop, Thomas A.R.Purcell, Matthias Scheffler, Christian Carbogno.

        Args:
            type (str): Specify fingerprint type needed can accept '{s/p/d/f/}summed_{pdos/tdos}'
            (default is summed_pdos)
            binning (bool): If true, the DOS fingerprint is binned using np.linspace and n_bins.
                Default is True.
            min_e (float): The minimum mode energy to include in the fingerprint (default is None)
            max_e (float): The maximum mode energy to include in the fingerprint (default is None)
            n_bins (int): Number of bins to be used in the fingerprint (default is 256)
            normalize (bool): If true, normalizes the area under fp to equal to 1. Default is True.

        Raises:
            ValueError: If type is not one of the accepted values {s/p/d/f/}summed_{pdos/tdos}.

        Returns:
            Fingerprint(namedtuple) : The electronic density of states fingerprint
                of format (energies, densities, type, n_bins)
        """
    fingerprint = namedtuple('fingerprint', 'energies densities type n_bins bin_width')
    energies = self.energies - self.efermi
    if max_e is None:
        max_e = np.max(energies)
    if min_e is None:
        min_e = np.min(energies)
    pdos_obj = self.get_spd_dos()
    pdos = {}
    for key in pdos_obj:
        dens = pdos_obj[key].get_densities()
        pdos[key.name] = dens
    pdos['summed_pdos'] = np.sum(list(pdos.values()), axis=0)
    pdos['tdos'] = self.get_densities()
    try:
        densities = pdos[type]
        if len(energies) < n_bins:
            inds = np.where((energies >= min_e) & (energies <= max_e))
            return fingerprint(energies[inds], densities[inds], type, len(energies), np.diff(energies)[0])
        if binning:
            ener_bounds = np.linspace(min_e, max_e, n_bins + 1)
            ener = ener_bounds[:-1] + (ener_bounds[1] - ener_bounds[0]) / 2.0
            bin_width = np.diff(ener)[0]
        else:
            ener_bounds = np.array(energies)
            ener = np.append(energies, [energies[-1] + np.abs(energies[-1]) / 10])
            n_bins = len(energies)
            bin_width = np.diff(energies)[0]
        dos_rebin = np.zeros(ener.shape)
        for ii, e1, e2 in zip(range(len(ener)), ener_bounds[0:-1], ener_bounds[1:]):
            inds = np.where((energies >= e1) & (energies < e2))
            dos_rebin[ii] = np.sum(densities[inds])
        if normalize:
            area = np.sum(dos_rebin * bin_width)
            dos_rebin_sc = dos_rebin / area
        else:
            dos_rebin_sc = dos_rebin
        return fingerprint(np.array([ener]), dos_rebin_sc, type, n_bins, bin_width)
    except KeyError:
        raise ValueError("Please recheck type requested, either the orbital projections unavailable in input DOS or there's a typo in type.")