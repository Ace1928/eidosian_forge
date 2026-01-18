import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def equivalent_lattice_points(self, uvw):
    """Return all lattice points equivalent to any of the lattice points
        in `uvw` with respect to rotations only.

        Only equivalent lattice points that conserves the distance to
        origo are included in the output (making this a kind of real
        space version of the equivalent_reflections() method).

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.equivalent_lattice_points([[0, 0, 2]])
        array([[ 0,  0, -2],
               [ 0, -2,  0],
               [-2,  0,  0],
               [ 2,  0,  0],
               [ 0,  2,  0],
               [ 0,  0,  2]])

        """
    uvw = np.array(uvw, ndmin=2)
    rot = self.get_rotations()
    n, nrot = (len(uvw), len(rot))
    directions = np.dot(uvw, rot).reshape((n * nrot, 3))
    ind = np.lexsort(directions.T)
    directions = directions[ind]
    diff = np.diff(directions, axis=0)
    mask = np.any(diff, axis=1)
    return np.vstack((directions[:-1][mask], directions[-1:]))