import operator
import math
def husl_to_rgb(h, s, l):
    return lch_to_rgb(*husl_to_lch([h, s, l]))