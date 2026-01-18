import sys
from . import links, tangles
def punct_torus():
    a = Crossing('a')
    a[0] = a[2]
    a[1] = a[3]
    return Link([a], check_planarity=False)