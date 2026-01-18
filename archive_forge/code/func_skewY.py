from math import cos, sin, tan, radians
def skewY(angle):
    return (1, tan(radians(angle)), 0, 1, 0, 0)