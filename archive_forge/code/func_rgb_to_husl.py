import operator
import math
def rgb_to_husl(r, g, b):
    return lch_to_husl(rgb_to_lch(r, g, b))