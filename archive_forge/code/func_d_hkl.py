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
def d_hkl(self, miller_index: ArrayLike) -> float:
    """Returns the distance between the hkl plane and the origin.

        Args:
            miller_index ([h,k,l]): Miller index of plane

        Returns:
            d_hkl (float)
        """
    g_star = self.reciprocal_lattice_crystallographic.metric_tensor
    hkl = np.array(miller_index)
    return 1 / np.dot(np.dot(hkl, g_star), hkl.T) ** (1 / 2)