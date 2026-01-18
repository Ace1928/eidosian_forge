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
@typing.no_type_check
@staticmethod
def from_origin_axis_angle(origin: ArrayLike, axis: ArrayLike, angle: float, angle_in_radians: bool=False) -> SymmOp:
    """Generates a SymmOp for a rotation about a given axis through an
        origin.

        Args:
            origin (3x1 array): The origin which the axis passes through.
            axis (3x1 array): The axis of rotation in Cartesian space. For
                example, [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.

        Returns:
            SymmOp.
        """
    theta = angle * pi / 180 if not angle_in_radians else angle
    a, b, c = origin
    ax_u, ax_v, ax_w = axis
    u2, v2, w2 = (ax_u * ax_u, ax_v * ax_v, ax_w * ax_w)
    cos_t = cos(theta)
    sin_t = sin(theta)
    l2 = u2 + v2 + w2
    lsqrt = sqrt(l2)
    m11 = (u2 + (v2 + w2) * cos_t) / l2
    m12 = (ax_u * ax_v * (1 - cos_t) - ax_w * lsqrt * sin_t) / l2
    m13 = (ax_u * ax_w * (1 - cos_t) + ax_v * lsqrt * sin_t) / l2
    m14 = (a * (v2 + w2) - ax_u * (b * ax_v + c * ax_w) + (ax_u * (b * ax_v + c * ax_w) - a * (v2 + w2)) * cos_t + (b * ax_w - c * ax_v) * lsqrt * sin_t) / l2
    m21 = (ax_u * ax_v * (1 - cos_t) + ax_w * lsqrt * sin_t) / l2
    m22 = (v2 + (u2 + w2) * cos_t) / l2
    m23 = (ax_v * ax_w * (1 - cos_t) - ax_u * lsqrt * sin_t) / l2
    m24 = (b * (u2 + w2) - ax_v * (a * ax_u + c * ax_w) + (ax_v * (a * ax_u + c * ax_w) - b * (u2 + w2)) * cos_t + (c * ax_u - a * ax_w) * lsqrt * sin_t) / l2
    m31 = (ax_u * ax_w * (1 - cos_t) - ax_v * lsqrt * sin_t) / l2
    m32 = (ax_v * ax_w * (1 - cos_t) + ax_u * lsqrt * sin_t) / l2
    m33 = (w2 + (u2 + v2) * cos_t) / l2
    m34 = (c * (u2 + v2) - ax_w * (a * ax_u + b * ax_v) + (ax_w * (a * ax_u + b * ax_v) - c * (u2 + v2)) * cos_t + (a * ax_v - b * ax_u) * lsqrt * sin_t) / l2
    return SymmOp([[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [0, 0, 0, 1]])