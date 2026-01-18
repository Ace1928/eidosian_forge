from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def get_factors(n):
    """Generate all factors of n."""
    for x in range(1, n + 1):
        if n % x == 0:
            yield x