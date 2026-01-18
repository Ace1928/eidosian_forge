from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def rel_strain(vec1, vec2):
    """Calculate relative strain between two vectors."""
    return fast_norm(vec2) / fast_norm(vec1) - 1