import numpy as np
def buchheim(tree):
    dt = first_walk(DrawTree(tree))
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt