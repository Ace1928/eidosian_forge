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
def b1(self):
    """Returns the derivative of bulk modulus w.r.t. pressure(dimensionless)."""
    return self._params[2]