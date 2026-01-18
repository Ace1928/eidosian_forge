from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def get_integer_index(miller_index: Sequence[float], round_dp: int=4, verbose: bool=True) -> tuple[int, int, int]:
    """Attempt to convert a vector of floats to whole numbers.

    Args:
        miller_index (list of float): A list miller indexes.
        round_dp (int, optional): The number of decimal places to round the
            miller index to.
        verbose (bool, optional): Whether to print warnings.

    Returns:
        tuple: The Miller index.
    """
    mi = np.asarray(miller_index)
    mi /= min((m for m in mi if m != 0))
    mi /= np.max(np.abs(mi))
    md = [Fraction(n).limit_denominator(12).denominator for n in mi]
    mi *= reduce(operator.mul, md)
    int_miller_index = np.round(mi, 1).astype(int)
    mi /= np.abs(reduce(math.gcd, int_miller_index))
    mi = np.array([round(h, round_dp) for h in mi])
    int_miller_index = np.round(mi, 1).astype(int)
    if np.any(np.abs(mi - int_miller_index) > 1e-06) and verbose:
        warnings.warn('Non-integer encountered in Miller index')
    else:
        mi = int_miller_index
    mi += 0

    def n_minus(index):
        return len([h for h in index if h < 0])
    if n_minus(mi) > n_minus(mi * -1):
        mi *= -1
    if sum(mi != 0) == 2 and n_minus(mi) == 1 and (abs(min(mi)) > max(mi)):
        mi *= -1
    return tuple(mi)