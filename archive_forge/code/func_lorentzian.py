from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from monty.json import MSONable
from scipy import stats
from scipy.ndimage import convolve1d
from pymatgen.util.coord import get_linear_interpolated_value
def lorentzian(x, x_0: float=0, sigma: float=1.0):
    """

    Args:
        x: x values
        x_0: Center
        sigma: FWHM.

    Returns:
        Value of lorentzian at x.
    """
    return 1 / np.pi * 0.5 * sigma / ((x - x_0) ** 2 + (0.5 * sigma) ** 2)