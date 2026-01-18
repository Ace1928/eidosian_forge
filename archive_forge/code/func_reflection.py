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
@staticmethod
def reflection(normal: ArrayLike, origin: ArrayLike=(0, 0, 0)) -> SymmOp:
    """Returns reflection symmetry operation.

        Args:
            normal (3x1 array): Vector of the normal to the plane of
                reflection.
            origin (3x1 array): A point in which the mirror plane passes
                through.

        Returns:
            SymmOp for the reflection about the plane
        """
    n = np.array(normal, dtype=float) / np.linalg.norm(normal)
    u, v, w = n
    translation = np.eye(4)
    translation[0:3, 3] = -np.array(origin)
    xx = 1 - 2 * u ** 2
    yy = 1 - 2 * v ** 2
    zz = 1 - 2 * w ** 2
    xy = -2 * u * v
    xz = -2 * u * w
    yz = -2 * v * w
    mirror_mat = [[xx, xy, xz, 0], [xy, yy, yz, 0], [xz, yz, zz, 0], [0, 0, 0, 1]]
    if np.linalg.norm(origin) > 1e-06:
        mirror_mat = np.dot(np.linalg.inv(translation), np.dot(mirror_mat, translation))
    return SymmOp(mirror_mat)