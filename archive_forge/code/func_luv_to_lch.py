import operator
import math
def luv_to_lch(triple):
    L, U, V = triple
    C = math.pow(math.pow(U, 2) + math.pow(V, 2), 1.0 / 2.0)
    hrad = math.atan2(V, U)
    H = math.degrees(hrad)
    if H < 0.0:
        H = 360.0 + H
    return [L, C, H]