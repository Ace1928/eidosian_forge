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
def get_distance_and_image(self, frac_coords1: ArrayLike, frac_coords2: ArrayLike, jimage: ArrayLike | None=None) -> tuple[float, np.ndarray]:
    """Gets distance between two frac_coords assuming periodic boundary
        conditions. If the index jimage is not specified it selects the j
        image nearest to the i atom and returns the distance and jimage
        indices in terms of lattice vector translations. If the index jimage
        is specified it returns the distance between the frac_coords1 and
        the specified jimage of frac_coords2, and the given jimage is also
        returned.

        Args:
            frac_coords1 (3x1 array): Reference frac_coords to get distance from.
            frac_coords2 (3x1 array): frac_coords to get distance from.
            jimage (3x1 array): Specific periodic image in terms of
                lattice translations, e.g., [1,0,0] implies to take periodic
                image that is one a-lattice vector away. If jimage is None,
                the image that is nearest to the site is found.

        Returns:
            tuple[float, np.ndarray]: distance and periodic lattice translations (jimage)
                of the other site for which the distance applies. This means that
                the distance between frac_coords1 and (jimage + frac_coords2) is
                equal to distance.
        """
    if jimage is None:
        v, d2 = pbc_shortest_vectors(self, frac_coords1, frac_coords2, return_d2=True)
        fc = self.get_fractional_coords(v[0][0]) + frac_coords1 - frac_coords2
        fc = np.array(np.round(fc), dtype=int)
        return (np.sqrt(d2[0, 0]), fc)
    jimage = np.array(jimage)
    mapped_vec = self.get_cartesian_coords(jimage + frac_coords2 - frac_coords1)
    return (np.linalg.norm(mapped_vec), jimage)