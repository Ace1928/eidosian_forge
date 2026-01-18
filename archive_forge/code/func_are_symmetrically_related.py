from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
def are_symmetrically_related(self, point_a: ArrayLike, point_b: ArrayLike, tol: float=0.001) -> bool:
    """Checks if two points are symmetrically related.

        Args:
            point_a (3x1 array): First point.
            point_b (3x1 array): Second point.
            tol (float): Absolute tolerance for checking distance. Defaults to 0.001.

        Returns:
            bool: True if self.operate(point_a) == point_b or vice versa.
        """
    return any((np.allclose(self.operate(p1), p2, atol=tol) for p1, p2 in [(point_a, point_b), (point_b, point_a)]))