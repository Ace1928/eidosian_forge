from __future__ import annotations
import os
import re
import warnings
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.analysis.excitation import ExcitationSpectrum
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Energy, FloatWithUnit
def get_excitation_spectrum(self, width=0.1, npoints=2000):
    """
        Generate an excitation spectra from the singlet roots of TDDFT calculations.

        Args:
            width (float): Width for Gaussian smearing.
            npoints (int): Number of energy points. More points => smoother
                curve.

        Returns:
            ExcitationSpectrum: can be plotted using pymatgen.vis.plotters.SpectrumPlotter.
        """
    roots = self.parse_tddft()
    data = roots['singlet']
    en = np.array([d['energy'] for d in data])
    osc = np.array([d['osc_strength'] for d in data])
    epad = 20.0 * width
    emin = en[0] - epad
    emax = en[-1] + epad
    de = (emax - emin) / npoints
    if width < 2 * de:
        width = 2 * de
    energies = [emin + ie * de for ie in range(npoints)]
    cutoff = 20.0 * width
    gamma = 0.5 * width
    gamma_sqrd = gamma * gamma
    de = (energies[-1] - energies[0]) / (len(energies) - 1)
    prefac = gamma / np.pi * de
    x = []
    y = []
    for energy in energies:
        xx0 = energy - en
        stot = osc / (xx0 * xx0 + gamma_sqrd)
        t = np.sum(stot[np.abs(xx0) <= cutoff])
        x.append(energy)
        y.append(t * prefac)
    return ExcitationSpectrum(x, y)