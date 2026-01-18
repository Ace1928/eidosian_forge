import operator
import math
def lch_to_huslp(triple):
    L, C, H = triple
    if L > 99.9999999:
        return [H, 0.0, 100.0]
    if L < 1e-08:
        return [H, 0.0, 0.0]
    mx = max_chroma_pastel(L)
    S = C / mx * 100.0
    return [H, S, L]