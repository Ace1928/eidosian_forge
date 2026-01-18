from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import leastsq, minimize
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
@property
def b0_GPa(self):
    """
        Returns the bulk modulus in GPa.
        Note: This assumes that the energy and volumes are in eV and Ang^3
            respectively.
        """
    return FloatWithUnit(self.b0, 'eV ang^-3').to('GPa')