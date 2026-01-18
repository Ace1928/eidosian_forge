from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def get_spectre_plot(self, sigma=0.05, step=0.01):
    """
        Get a matplotlib plot of the UV-visible xas. Transitions are plotted
        as vertical lines and as a sum of normal functions with sigma with. The
        broadening is applied in energy and the xas is plotted as a function
        of the wavelength.

        Args:
            sigma: Full width at half maximum in eV for normal functions.
            step: bin interval in eV

        Returns:
            A dict: {"energies": values, "lambda": values, "xas": values}
                    where values are lists of abscissa (energies, lamba) and
                    the sum of gaussian functions (xas).
            A matplotlib plot.
        """
    ax = pretty_plot(12, 8)
    transitions = self.read_excitation_energies()
    minval = min((val[0] for val in transitions)) - 5.0 * sigma
    maxval = max((val[0] for val in transitions)) + 5.0 * sigma
    npts = int((maxval - minval) / step) + 1
    eneval = np.linspace(minval, maxval, npts)
    lambdaval = [cst.h * cst.c / (val * cst.e) * 1000000000.0 for val in eneval]
    spectre = np.zeros(npts)
    for trans in transitions:
        spectre += trans[2] * norm(eneval, trans[0], sigma)
    spectre /= spectre.max()
    ax.plot(lambdaval, spectre, 'r-', label='spectre')
    data = {'energies': eneval, 'lambda': lambdaval, 'xas': spectre}
    ax.vlines([val[1] for val in transitions], 0.0, [val[2] for val in transitions], color='blue', label='transitions', linewidth=2)
    ax.set_xlabel('$\\lambda$ (nm)')
    ax.set_ylabel('Arbitrary unit')
    ax.legend()
    return (data, ax)