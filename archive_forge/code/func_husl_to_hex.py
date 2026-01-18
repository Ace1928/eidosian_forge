import operator
import math
def husl_to_hex(h, s, l):
    return rgb_to_hex(husl_to_rgb(h, s, l))