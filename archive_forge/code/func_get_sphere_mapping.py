import pyglet.gl as pgl
from math import sqrt as _sqrt, acos as _acos
def get_sphere_mapping(x, y, width, height):
    x = min([max([x, 0]), width])
    y = min([max([y, 0]), height])
    sr = _sqrt((width / 2) ** 2 + (height / 2) ** 2)
    sx = (x - width / 2) / sr
    sy = (y - height / 2) / sr
    sz = 1.0 - sx ** 2 - sy ** 2
    if sz > 0.0:
        sz = _sqrt(sz)
        return (sx, sy, sz)
    else:
        sz = 0
        return norm((sx, sy, sz))