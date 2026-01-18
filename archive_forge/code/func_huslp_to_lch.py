import operator
import math
def huslp_to_lch(triple):
    H, S, L = triple
    if L > 99.9999999:
        return [100, 0.0, H]
    if L < 1e-08:
        return [0.0, 0.0, H]
    mx = max_chroma_pastel(L)
    C = mx / 100.0 * S
    return [L, C, H]