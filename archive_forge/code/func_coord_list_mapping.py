from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def coord_list_mapping(subset: ArrayLike, superset: ArrayLike, atol: float=1e-08):
    """Gives the index mapping from a subset to a superset.
    Subset and superset cannot contain duplicate rows.

    Args:
        subset (ArrayLike): List of coords
        superset (ArrayLike): List of coords
        atol (float): Absolute tolerance. Defaults to 1e-8.

    Returns:
        list of indices such that superset[indices] = subset
    """
    c1 = np.array(subset)
    c2 = np.array(superset)
    inds = np.where(np.all(np.isclose(c1[:, None, :], c2[None, :, :], atol=atol), axis=2))[1]
    result = c2[inds]
    if not np.allclose(c1, result, atol=atol) and (not is_coord_subset(subset, superset)):
        raise ValueError('not a subset of superset')
    if not result.shape == c1.shape:
        raise ValueError('Something wrong with the inputs, likely duplicates in superset')
    return inds