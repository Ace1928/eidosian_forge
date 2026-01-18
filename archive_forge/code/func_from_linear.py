import operator
import math
def from_linear(c):
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * math.pow(c, 1.0 / 2.4) - 0.055