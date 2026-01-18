import numpy as np
def second_walk(v, m=0, depth=0, min=None):
    v.x += m
    v.y = depth
    if min is None or v.x < min:
        min = v.x
    for w in v.children:
        min = second_walk(w, m + v.mod, depth + 1, min)
    return min