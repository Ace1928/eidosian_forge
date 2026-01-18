import operator
import math
def lch_to_luv(triple):
    L, C, H = triple
    Hrad = math.radians(H)
    U = math.cos(Hrad) * C
    V = math.sin(Hrad) * C
    return [L, U, V]