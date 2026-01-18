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
@classmethod
def monoclinic(cls, a: float, b: float, c: float, beta: float, pbc: PbcLike=(True, True, True)) -> Self:
    """Convenience constructor for a monoclinic lattice.

        Args:
            a (float): *a* lattice parameter of the monoclinic cell.
            b (float): *b* lattice parameter of the monoclinic cell.
            c (float): *c* lattice parameter of the monoclinic cell.
            beta (float): *beta* angle between lattice vectors b and c in
                degrees.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Monoclinic lattice of dimensions a x b x c with non right-angle
            beta between lattice vectors a and c.
        """
    return cls.from_parameters(a, b, c, 90, beta, 90, pbc=pbc)