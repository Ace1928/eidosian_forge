from __future__ import annotations
import os
from glob import glob
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable, jsanitize
from scipy.interpolate import CubicSpline
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar
from pymatgen.util.plotting import pretty_plot
def setup_spline(self, spline_options=None):
    """
        Setup of the options for the spline interpolation.

        Args:
            spline_options (dict): Options for cubic spline. For example,
                {"saddle_point": "zero_slope"} forces the slope at the saddle to
                be zero.
        """
    self.spline_options = spline_options
    relative_energies = self.energies - self.energies[0]
    if self.spline_options.get('saddle_point', '') == 'zero_slope':
        imax = np.argmax(relative_energies)
        self.spline = CubicSpline(x=self.r[:imax + 1], y=relative_energies[:imax + 1], bc_type=((1, 0.0), (1, 0.0)))
        cspline2 = CubicSpline(x=self.r[imax:], y=relative_energies[imax:], bc_type=((1, 0.0), (1, 0.0)))
        self.spline.extend(c=cspline2.c, x=cspline2.x[1:])
    else:
        self.spline = CubicSpline(x=self.r, y=relative_energies, bc_type=((1, 0.0), (1, 0.0)))