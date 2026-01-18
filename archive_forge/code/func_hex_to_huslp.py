import operator
import math
def hex_to_huslp(hex):
    return rgb_to_huslp(*hex_to_rgb(hex))