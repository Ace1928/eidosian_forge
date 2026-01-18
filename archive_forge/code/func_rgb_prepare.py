import operator
import math
def rgb_prepare(triple):
    ret = []
    for ch in triple:
        ch = round(ch, 3)
        if ch < -0.0001 or ch > 1.0001:
            raise Exception(f'Illegal RGB value {ch:f}')
        if ch < 0:
            ch = 0
        if ch > 1:
            ch = 1
        ret.append(int(round(ch * 255 + 0.001, 0)))
    return ret