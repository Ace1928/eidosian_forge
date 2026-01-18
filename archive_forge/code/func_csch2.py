from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def csch2(x):
    return 1.0 / np.sinh(x) ** 2