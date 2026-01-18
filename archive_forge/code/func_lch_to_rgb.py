import operator
import math
def lch_to_rgb(l, c, h):
    return xyz_to_rgb(luv_to_xyz(lch_to_luv([l, c, h])))