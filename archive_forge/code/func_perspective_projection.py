from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@classmethod
def perspective_projection(cls: type[Mat4T], aspect: float, z_near: float, z_far: float, fov: float=60) -> Mat4T:
    """Create a Mat4 perspective projection matrix for use with OpenGL.

        Given a desired aspect ratio, near/far planes, and fov (field of view),
        create a 4x4 Projection Matrix. This is useful for setting
        :py:attr:`~pyglet.window.Window.projection`.
        """
    xy_max = z_near * _math.tan(fov * _math.pi / 360)
    y_min = -xy_max
    x_min = -xy_max
    width = xy_max - x_min
    height = xy_max - y_min
    depth = z_far - z_near
    q = -(z_far + z_near) / depth
    qn = -2 * z_far * z_near / depth
    w = 2 * z_near / width
    w = w / aspect
    h = 2 * z_near / height
    return cls((w, 0, 0, 0, 0, h, 0, 0, 0, 0, q, -1, 0, 0, qn, 0))