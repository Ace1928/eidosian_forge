from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@classmethod
def look_at(cls: type[Mat4T], position: Vec3, target: Vec3, up: Vec3):
    f = (target - position).normalize()
    u = up.normalize()
    s = f.cross(u).normalize()
    u = s.cross(f)
    return cls([s.x, u.x, -f.x, 0.0, s.y, u.y, -f.y, 0.0, s.z, u.z, -f.z, 0.0, -s.dot(position), -u.dot(position), f.dot(position), 1.0])