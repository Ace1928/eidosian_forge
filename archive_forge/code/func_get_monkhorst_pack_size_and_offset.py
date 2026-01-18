import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def get_monkhorst_pack_size_and_offset(kpts):
    """Find Monkhorst-Pack size and offset.

    Returns (size, offset), where::

        kpts = monkhorst_pack(size) + offset.

    The set of k-points must not have been symmetry reduced."""
    if len(kpts) == 1:
        return (np.ones(3, int), np.array(kpts[0], dtype=float))
    size = np.zeros(3, int)
    for c in range(3):
        delta = max(np.diff(np.sort(kpts[:, c])))
        if delta > 1e-08:
            size[c] = int(round(1.0 / delta))
        else:
            size[c] = 1
    if size.prod() == len(kpts):
        kpts0 = monkhorst_pack(size)
        offsets = kpts - kpts0
        if (offsets.ptp(axis=0) < 1e-09).all():
            return (size, offsets[0].copy())
    raise ValueError('Not an ASE-style Monkhorst-Pack grid!')