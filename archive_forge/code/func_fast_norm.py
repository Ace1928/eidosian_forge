from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def fast_norm(a):
    """
    Much faster variant of numpy linalg norm.

    Note that if numba is installed, this cannot be provided a list of ints;
    please ensure input a is an np.array of floats.
    """
    return np.sqrt(np.dot(a, a))