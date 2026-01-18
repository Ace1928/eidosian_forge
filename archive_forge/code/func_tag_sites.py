import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def tag_sites(self, scaled_positions, symprec=0.001):
    """Returns an integer array of the same length as *scaled_positions*,
        tagging all equivalent atoms with the same index.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.tag_sites([[0.0, 0.0, 0.0],
        ...               [0.5, 0.5, 0.0],
        ...               [1.0, 0.0, 0.0],
        ...               [0.5, 0.0, 0.0]])
        array([0, 0, 0, 1])
        """
    scaled = np.array(scaled_positions, ndmin=2)
    scaled %= 1.0
    scaled %= 1.0
    tags = -np.ones((len(scaled),), dtype=int)
    mask = np.ones((len(scaled),), dtype=bool)
    rot, trans = self.get_op()
    i = 0
    while mask.any():
        pos = scaled[mask][0]
        sympos = np.dot(rot, pos) + trans
        sympos %= 1.0
        sympos %= 1.0
        m = ~np.all(np.any(np.abs(scaled[np.newaxis, :, :] - sympos[:, np.newaxis, :]) > symprec, axis=2), axis=0)
        assert not np.any(~mask & m)
        tags[m] = i
        mask &= ~m
        i += 1
    return tags