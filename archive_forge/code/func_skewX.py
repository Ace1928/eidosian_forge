from math import cos, sin, tan, radians
def skewX(angle):
    return (1, 0, tan(radians(angle)), 1, 0, 0)