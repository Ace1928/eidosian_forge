import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def labels_from_kpts(kpts, cell, eps=1e-05, special_points=None):
    """Get an x-axis to be used when plotting a band structure.

    The first of the returned lists can be used as a x-axis when plotting
    the band structure. The second list can be used as xticks, and the third
    as xticklabels.

    Parameters:

    kpts: list
        List of scaled k-points.

    cell: list
        Unit cell of the atomic structure.

    Returns:

    Three arrays; the first is a list of cumulative distances between k-points,
    the second is x coordinates of the special points,
    the third is the special points as strings.
    """
    if special_points is None:
        special_points = get_special_points(cell)
    points = np.asarray(kpts)
    indices = find_bandpath_kinks(cell, kpts, eps=1e-05)
    labels = []
    for kpt in points[indices]:
        for label, k in special_points.items():
            if abs(kpt - k).sum() < eps:
                break
        else:
            for label, k in special_points.items():
                if abs((kpt - k) % 1).sum() < eps:
                    break
            else:
                label = '?'
        labels.append(label)
    xcoords, ixcoords = indices_to_axis_coords(indices, points, cell)
    return (xcoords, ixcoords, labels)