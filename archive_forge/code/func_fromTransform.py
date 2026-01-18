import math
from typing import NamedTuple
from dataclasses import dataclass
@classmethod
def fromTransform(self, transform):
    a, b, c, d, x, y = transform
    sx = math.copysign(1, a)
    if sx < 0:
        a *= sx
        b *= sx
    delta = a * d - b * c
    rotation = 0
    scaleX = scaleY = 0
    skewX = skewY = 0
    if a != 0 or b != 0:
        r = math.sqrt(a * a + b * b)
        rotation = math.acos(a / r) if b >= 0 else -math.acos(a / r)
        scaleX, scaleY = (r, delta / r)
        skewX, skewY = (math.atan((a * c + b * d) / (r * r)), 0)
    elif c != 0 or d != 0:
        s = math.sqrt(c * c + d * d)
        rotation = math.pi / 2 - (math.acos(-c / s) if d >= 0 else -math.acos(c / s))
        scaleX, scaleY = (delta / s, s)
        skewX, skewY = (0, math.atan((a * c + b * d) / (s * s)))
    else:
        pass
    return DecomposedTransform(x, y, math.degrees(rotation), scaleX * sx, scaleY, math.degrees(skewX) * sx, math.degrees(skewY), 0, 0)