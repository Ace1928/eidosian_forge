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
def rotoreflection(axis: ArrayLike, angle: float, origin: ArrayLike=(0, 0, 0)) -> SymmOp:
    """Returns a roto-reflection symmetry operation.

        Args:
            axis (3x1 array): Axis of rotation / mirror normal
            angle (float): Angle in degrees
            origin (3x1 array): Point left invariant by roto-reflection.
                Defaults to (0, 0, 0).

        Returns:
            Roto-reflection operation
        """
    rot = SymmOp.from_origin_axis_angle(origin, axis, angle)
    refl = SymmOp.reflection(axis, origin)
    m = np.dot(rot.affine_matrix, refl.affine_matrix)
    return SymmOp(m)