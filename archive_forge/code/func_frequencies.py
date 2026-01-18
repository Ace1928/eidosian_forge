import math
def frequencies(it):
    c = {}
    for i in it:
        c[i] = c.get(i, 0) + 1
    return c